import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbones.ltae import LTAE2d

# Reuse your existing blocks
from .utae import (
    ConvBlock,
    DownConvBlock,
    Temporal_Aggregator,
)

class DeepLabV3PlusTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 128, 256], # [Stride 1/2, Stride 4, Stride 8, Stride 16]
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
        low_level_idx=1, # Index in encoder_widths to use as low-level feature (usually stride 4)
        aspp_rates=[6, 12, 18],
    ):
        super(DeepLabV3PlusTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.pad_value = pad_value
        self.encoder = encoder
        self.low_level_idx = low_level_idx
        
        # --- ENCODER ---
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )

        # --- TEMPORAL BOTTLENECK ---
        # LTAE takes the High Level Feature (Deepest)
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        # Used for the skip connection
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # --- ASPP ---
        # Applied after LTAE (Time is collapsed)
        self.aspp = ASPP(
            in_channels=encoder_widths[-1],
            out_channels=256,
            rates=aspp_rates
        )

        # --- DECODER ---
        # 1. Projection for Low Level Feature
        low_level_channels = encoder_widths[low_level_idx]
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 2. Main Decoder Convolutions
        # Input = ASPP output (256) + Low Level Projected (48) = 304
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # --- OUT HEAD ---
        # Standard DeepLab usually outputs classes directly here, 
        # but to match UTAE signature we use out_conv block if provided, or simple Conv
        self.out_conv = ConvBlock(
            nkernels=[256] + out_conv, 
            padding_mode=padding_mode
        )

    def forward(self, input, batch_positions=None, return_att=False):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        
        # --- SPATIAL ENCODER ---
        out = self.in_conv.smart_forward(input)
        encoder_features = [out] # Stride 1 or 2
        
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(encoder_features[-1])
            encoder_features.append(out) 
            
        # encoder_features contains all maps.
        # High Level = encoder_features[-1]
        # Low Level = encoder_features[self.low_level_idx]

        # --- TEMPORAL BOTTLENECK ---
        # Collapses Time: (B, T, C, H, W) -> (B, C, H, W)
        high_level_feat, att = self.temporal_encoder(
            encoder_features[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )

        # --- ASPP ---
        x_aspp = self.aspp(high_level_feat) # (B, 256, H_high, W_high)

        # --- DECODER START ---
        # 1. Prepare Low Level Feature
        low_level_feat = encoder_features[self.low_level_idx] # (B, T, C_low, H_low, W_low)
        
        # Aggregate Low Level Feature using LTAE Attention Mask
        # We re-use the attention calculated at the bottleneck
        low_level_feat = self.temporal_aggregator(
            low_level_feat, pad_mask=pad_mask, attn_mask=att
        ) # (B, C_low, H_low, W_low)
        
        # Project 1x1
        low_level_feat = self.low_level_conv(low_level_feat) # (B, 48, H_low, W_low)

        # 2. Upsample ASPP to Low Level Resolution
        x_aspp_up = F.interpolate(
            x_aspp, 
            size=low_level_feat.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        # 3. Concatenate
        dec_in = torch.cat([x_aspp_up, low_level_feat], dim=1)

        # 4. Refine
        dec_out = self.decoder_conv(dec_in)

        # --- OUTPUT ---
        if self.encoder:
             return dec_out, [dec_out] # Format to match UTAE signature
        
        # Final convs
        out = self.out_conv(dec_out)
        
        # Final Upsample to input resolution
        out = F.interpolate(
            out, 
            size=input.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        if return_att:
            return out, att
        if self.return_maps:
            return out, [dec_out]
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        modules = []
        # 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Dilated Convs
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global Avg Pooling (Image Level Feature)
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)

        # Projection after concatenation
        # Input channels = (len(rates) + 1_1x1 + 1_ImagePooling) * out_channels
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # Check if it's the Image Pooling layer (last one in list with AdaptivePool)
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                gap = conv(x)
                # Upsample GAP back to feature size
                gap = F.interpolate(gap, size=x.shape[-2:], mode='bilinear', align_corners=False)
                res.append(gap)
            else:
                res.append(conv(x))
        
        res = torch.cat(res, dim=1)
        return self.project(res)