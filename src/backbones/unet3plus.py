import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.backbones.ltae import LTAE2d


class ConvLayer(nn.Module):
    def __init__(self, nkernels, norm="batch", k=3, s=1, p=1, n_groups=4, last_relu=True, padding_mode="reflect"):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(nn.Conv2d(in_channels=nkernels[i], out_channels=nkernels[i + 1], kernel_size=k, padding=p, stride=s, padding_mode=padding_mode))
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            if last_relu:
                layers.append(nn.ReLU(inplace=True))
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class TemporallySharedBlock(nn.Module):
    """
    Apply the same 2D block to inputs that are either 4D (B,C,H,W) or 5D (B,T,C,H,W).
    This implementation robustly handles pad_value: it computes a mask over the merged
    B*T frames and processes only valid frames to save compute and avoid indexing errors.
    """
    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if input.dim() == 4:
            return self.forward(input)

        b, t, c, h, w = input.shape
        merged = input.view(b * t, c, h, w)  # (B*T, C, H, W)

        if self.pad_value is None:
            out_merged = self.forward(merged)
            _, c2, h2, w2 = out_merged.shape
            return out_merged.view(b, t, c2, h2, w2)

        # Compute per-frame pad mask aligned with merged
        pad_frame_mask = (merged == self.pad_value).all(dim=(1, 2, 3))  # (B*T,)

        # If all frames are padded: return full pad_value output with proper shape
        if pad_frame_mask.all():
            if self.out_shape is None:
                # infer out shape by running dummy
                dummy = torch.zeros_like(merged)
                tmp = self.forward(dummy)
                self.out_shape = tmp.shape
            c2, h2, w2 = self.out_shape[1], self.out_shape[2], self.out_shape[3]
            return torch.full((b, t, c2, h2, w2), fill_value=self.pad_value, device=input.device, dtype=merged.dtype)

        valid_idx = (~pad_frame_mask).nonzero(as_tuple=False).squeeze(1)
        valid_merged = merged[valid_idx]

        out_valid = self.forward(valid_merged)

        if self.out_shape is None:
            self.out_shape = out_valid.shape

        N_total = b * t
        C_out, H_out, W_out = out_valid.shape[1], out_valid.shape[2], out_valid.shape[3]
        device = out_valid.device
        dtype = out_valid.dtype

        out_merged_full = torch.full((N_total, C_out, H_out, W_out), fill_value=self.pad_value, device=device, dtype=dtype)
        out_merged_full[valid_idx] = out_valid

        out = out_merged_full.view(b, t, C_out, H_out, W_out)
        return out


class ConvBlock(TemporallySharedBlock):
    def __init__(self, nkernels, pad_value=None, norm="batch", last_relu=True, padding_mode="reflect"):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(nkernels=nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode)

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(self, d_in, d_out, k, s, p, pad_value=None, norm="batch", padding_mode="reflect"):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(nkernels=[d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = ConvLayer(nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p), nn.BatchNorm2d(d_out), nn.ReLU(inplace=True))
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


# -----------------------------
# Temporal aggregator (supports att_group, att_mean, mean)
# -----------------------------
class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="att_group"):
        super(Temporal_Aggregator, self).__init__()
        assert mode in ("att_group", "att_mean", "mean")
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        # x: (B, T, C, H, W)
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                # attn_mask: n_head x B x T x H_att x W_att
                n_heads, b, t, h_att, w_att = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h_att, w_att)

                if x.shape[-2] > h_att:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                else:
                    pool_k = max(1, w_att // x.shape[-2])
                    attn = nn.AvgPool2d(kernel_size=pool_k)(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC_hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC_hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # B x T x H_att x W_att
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h_att, w_att = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h_att, w_att)
                if x.shape[-2] > h_att:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                else:
                    pool_k = max(1, w_att // x.shape[-2])
                    attn = nn.AvgPool2d(kernel_size=pool_k)(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)
                out = torch.cat([group for group in out], dim=1)
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


# -----------------------------
# Full UNet3+ encoder/decoder with multi-level LTAE
# -----------------------------
class UNet3P_Encoder(nn.Module):
    def __init__(self, in_ch, widths, pad_value=None, norm="group"):
        super(UNet3P_Encoder, self).__init__()
        self.n_levels = len(widths)
        self.pad_value = pad_value
        self.in_conv = ConvBlock(nkernels=[in_ch, widths[0], widths[0]], pad_value=pad_value, norm=norm)
        self.down_blocks = nn.ModuleList()
        for i in range(self.n_levels - 1):
            self.down_blocks.append(DownConvBlock(d_in=widths[i], d_out=widths[i + 1], k=4, s=2, p=1, pad_value=pad_value, norm=norm))

    def forward(self, x):
        # x can be (B,C,H,W) or (B,T,C,H,W)
        out = self.in_conv.smart_forward(x)
        feats = [out]
        for blk in self.down_blocks:
            out = blk.smart_forward(feats[-1])
            feats.append(out)
        # feats: list length n_levels, each either (B,C,H,W) or (B,T,C,H,W)
        return feats


class UNet3P_DecoderStage(nn.Module):
    """
    Decoder stage that fuses multi-scale features as in UNet3+. For simplicity we implement a fusion by
    projecting each input feature to a common channel count and up/downsampling to the target resolution.
    """
    def __init__(self, widths, target_idx, proj_ch, padding_mode="reflect"):
        super(UNet3P_DecoderStage, self).__init__()
        self.target_idx = target_idx
        self.n_levels = len(widths)
        self.proj_convs = nn.ModuleList()
        for i, w in enumerate(widths):
            # projection conv to proj_ch
            self.proj_convs.append(nn.Sequential(nn.Conv2d(w, proj_ch, kernel_size=1), nn.BatchNorm2d(proj_ch), nn.ReLU(inplace=True)))
        # after concatenation
        self.fuse = ConvLayer(nkernels=[proj_ch * self.n_levels, proj_ch], norm="batch")

    def forward(self, feats):
        # feats: list of tensors at different scales; each tensor is (B, C_i, H_i, W_i)
        target = feats[self.target_idx]
        Ht, Wt = target.shape[-2], target.shape[-1]
        ups = []
        for i, f in enumerate(feats):
            p = self.proj_convs[i](f)
            if f.shape[-2] != Ht:
                p = F.interpolate(p, size=(Ht, Wt), mode='bilinear', align_corners=False)
            ups.append(p)
        cat = torch.cat(ups, dim=1)
        return self.fuse(cat)


class UNet3P_Full(nn.Module):
    def __init__(self, input_dim=10, encoder_widths=[64,128,256,512,512], decoder_proj=128, num_classes=20, pad_value=0, n_head=16, d_model=256, d_k=4, agg_mode='att_group', deep_supervision=False):
        super(UNet3P_Full, self).__init__()
        self.pad_value = pad_value
        self.n_levels = len(encoder_widths)
        self.encoder = UNet3P_Encoder(in_ch=input_dim, widths=encoder_widths, pad_value=pad_value, norm="group")

        # Multi-level temporal encoders (LTAE) - one per encoder level
        self.temporal_encoders = nn.ModuleList()
        for i in range(self.n_levels):
            ch = encoder_widths[i]
            self.temporal_encoders.append(LTAE2d(in_channels=ch, d_model=d_model, n_head=n_head, mlp=[d_model, ch], return_att=True, d_k=d_k))

        # Temporal aggregator used for skips
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # Decoder stages - produce fused features at each level
        self.decoder_stages = nn.ModuleList()
        for i in range(self.n_levels):
            self.decoder_stages.append(UNet3P_DecoderStage(encoder_widths, target_idx=i, proj_ch=decoder_proj))

        # final conv head
        self.final_head = nn.Sequential(nn.Conv2d(decoder_proj, decoder_proj//2, 3, padding=1), nn.BatchNorm2d(decoder_proj//2), nn.ReLU(inplace=True), nn.Conv2d(decoder_proj//2, num_classes, 1))

        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.deep_heads = nn.ModuleList([nn.Conv2d(decoder_proj, num_classes, 1) for _ in range(self.n_levels)])

    def forward(self, x, batch_positions=None, return_att=False):
        """
        x: (B, T, C, H, W)
        batch_positions: (B, T) or None
        returns: (B, num_classes, H, W) or (out, att_dict) if return_att=True
        """
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # BxT

        # SPATIAL ENCODER (shared convs across time)
        feats = self.encoder.forward(x)  # list len n_levels, each (B,T,C_i,H_i,W_i)

        # TEMPORAL ENCODER applied at each level
        emb_per_level = []
        att_per_level = []
        for lvl, (feat, tenc) in enumerate(zip(feats, self.temporal_encoders)):
            # feat: (B,T,C,H,W)
            emb, att = tenc(feat, batch_positions=batch_positions, pad_mask=pad_mask)
            # emb: (B,C,H,W), att: (n_head, B, T, H_att, W_att)
            emb_per_level.append(emb)
            att_per_level.append(att)

        # TEMPORAL AGGREGATION for skips depends on the attn per level
        # Build list of aggregated skip features (same resolution as encoder features)
        skip_aggregated = []
        for lvl in range(self.n_levels):
            # for deepest level, emb_per_level already is the embedding
            if lvl == self.n_levels - 1:
                skip_aggregated.append(emb_per_level[lvl])
            else:
                agg = self.temporal_aggregator(feats[lvl], pad_mask=pad_mask, attn_mask=att_per_level[lvl])
                skip_aggregated.append(agg)

        # Now decode using UNet3+ style dense fusion but using aggregated skips
        # For UNet3+ fusion, each decoder stage fuses ALL skip_aggregated levels resized to target resolution.
        decoded_per_level = [None] * self.n_levels
        for i in range(self.n_levels - 1, -1, -1):
            # prepare list of feats at each level for fusion; use current available decoded_per_level entries if present
            fusion_inputs = []
            for lvl in range(self.n_levels):
                # base feature for level lvl: if we've decoded a finer representation for that level, use it; else use skip_aggregated
                base = skip_aggregated[lvl] if decoded_per_level[lvl] is None else decoded_per_level[lvl]
                fusion_inputs.append(base)
            # fuse to produce feature at level i
            decoded = self.decoder_stages[i](fusion_inputs)
            decoded_per_level[i] = decoded

        out = self.final_head(decoded_per_level[0])

        if return_att:
            att_dict = {f"att_level_{i}": att_per_level[i] for i in range(len(att_per_level))}
            return out, att_dict
        else:
            return out