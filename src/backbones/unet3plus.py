"""
UTAE-UNet3+ hybrid backbone for PASTIS semantic segmentation.

Model API:
    model = UTAE_UNet3P(input_dim=10, encoder_widths=[64,64,64,128], decoder_widths=[32,32,64,128], ...)
    out = model(x, batch_positions=dates)  # x: (B,T,C,H,W) -> out: (B,num_classes,H,W)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbones.ltae import LTAE2d


# -------------------------
# Reused building blocks
# -------------------------
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
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    This implementation carefully handles pad_value so the indexing shapes match.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        # If single-frame (B,C,H,W) just forward
        if input.dim() == 4:
            return self.forward(input)

        # input is (B, T, C, H, W)
        b, t, c, h, w = input.shape
        merged = input.view(b * t, c, h, w)  # (B*T, C, H, W)

        # If we don't use pad_value, just process everything
        if self.pad_value is None:
            out_merged = self.forward(merged)  # (B*T, C_out, H2, W2)
            c2, h2, w2 = out_merged.shape[1], out_merged.shape[2], out_merged.shape[3]
            return out_merged.view(b, t, c2, h2, w2)

        # Compute per-frame pad mask after merging
        # pad_frame_mask[i] == True if merged[i] is entirely padding (all channels and pixels equal pad_value)
        pad_frame_mask = (merged == self.pad_value).all(dim=(1, 2, 3))  # shape: (B*T,)

        # If no valid frames, return a tensor filled with pad_value (using the expected out shape)
        if pad_frame_mask.all():
            # Need to infer output shape by forwarding a dummy tensor
            if self.out_shape is None:
                # run forward on a dummy to get shape
                dummy = torch.zeros_like(merged)
                try:
                    dummy_out = self.forward(dummy)
                except Exception:
                    # If forward depends on data shape we still should create something reasonable:
                    # assume output channels same as input channels and same spatial size
                    c2, h2, w2 = c, h, w
                else:
                    self.out_shape = dummy_out.shape
                    c2, h2, w2 = self.out_shape[1], self.out_shape[2], self.out_shape[3]
            else:
                c2, h2, w2 = self.out_shape[1], self.out_shape[2], self.out_shape[3]

            fill = torch.full((b, t, c2, h2, w2), self.pad_value, device=input.device, dtype=merged.dtype)
            return fill

        # There are some valid frames. Process only them.
        valid_idx = (~pad_frame_mask).nonzero(as_tuple=False).squeeze(1)  # indices of valid frames
        valid_merged = merged[valid_idx]  # (N_valid, C, H, W)

        # Compute outputs for valid frames
        out_valid = self.forward(valid_merged)  # (N_valid, C_out, H2, W2)

        # Save output shape for potential future all-padded batches
        if self.out_shape is None:
            self.out_shape = out_valid.shape

        # Create full output merged tensor populated with pad_value then fill valid slots
        N_total = b * t
        C_out, H_out, W_out = out_valid.shape[1], out_valid.shape[2], out_valid.shape[3]
        dtype = out_valid.dtype
        device = out_valid.device
        out_merged_full = torch.full((N_total, C_out, H_out, W_out), fill_value=self.pad_value, device=device, dtype=dtype)

        out_merged_full[valid_idx] = out_valid

        # Reshape back to (B, T, C_out, H_out, W_out)
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
        self.skip_conv = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1), nn.BatchNorm2d(d), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p), nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


# -------------------------
# Temporal aggregator (copy of the UTAE implementation's aggregator)
# -------------------------
class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        # x: (B, T, C, H, W)
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
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
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


# -------------------------
# UTAE-UNet3+ hybrid model
# -------------------------
class UTAE_UNet3P(nn.Module):
    def __init__(
        self,
        input_dim=10,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=None,
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
    ):
        super(UTAE_UNet3P, self).__init__()
        self.n_stages = len(encoder_widths)
        self.pad_value = pad_value
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths if decoder_widths is not None else encoder_widths
        assert len(self.decoder_widths) == self.n_stages

        # input conv (shared across time)
        self.in_conv = ConvBlock(nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]], pad_value=pad_value, norm=encoder_norm)

        # down blocks (shared across time)
        self.down_blocks = nn.ModuleList([
            DownConvBlock(d_in=encoder_widths[i], d_out=encoder_widths[i + 1], k=str_conv_k, s=str_conv_s, p=str_conv_p, pad_value=pad_value, norm=encoder_norm)
            for i in range(self.n_stages - 1)
        ])

        # temporal encoder (on deepest level)
        self.temporal_encoder = LTAE2d(in_channels=encoder_widths[-1], d_model=d_model, n_head=n_head, mlp=[d_model, encoder_widths[-1]], return_att=True, d_k=d_k)

        # temporal aggregator for skip connections
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # up (decoder) blocks - UNet3+ like dense fusion simplified per stage
        self.up_blocks = nn.ModuleList([
            UpConvBlock(d_in=self.decoder_widths[i], d_out=self.decoder_widths[i - 1], k=str_conv_k, s=str_conv_s, p=str_conv_p, norm="batch", d_skip=encoder_widths[i - 1])
            for i in range(self.n_stages - 1, 0, -1)
        ])

        # final output conv
        self.out_conv = ConvBlock(nkernels=[self.decoder_widths[0]] + out_conv)

    def forward(self, input, batch_positions=None, return_att=False):
        """
        input: (B, T, C, H, W)
        batch_positions: (B, T) float/int tensor with acquisition times (used by LTAE2d positional encoder)
        returns: logits (B, num_classes, H, W)
        """
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # BxT pad mask

        # SPATIAL ENCODER: apply shared convolutions across time using smart_forward
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # feature_maps: list of length n_stages, each is (B, T, C_i, H_i, W_i)

        # TEMPORAL ENCODER on deepest level
        out, att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
        # out: (B, C_last, H_last, W_last)
        if return_att:
            att_out = att

        # SPATIAL DECODER: aggregate temporally each skip and decode
        maps = [out]
        for i in range(self.n_stages - 1):
            # skip from shallower level: feature_maps[-(i+2)] -> (B, T, C, H, W)
            skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att)
            # upsample previous out and fuse with skip via UpConvBlock
            out = self.up_blocks[i](out, skip)
            maps.append(out)

        # final classification conv
        out = self.out_conv(out)

        if return_att:
            return out, att_out
        else:
            return out