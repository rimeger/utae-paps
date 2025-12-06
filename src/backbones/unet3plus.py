import torch
import torch.nn as nn
from src.backbones.ltae import LTAE2d

# Reusing components from your snippet
from .utae import (
    ConvBlock,
    ConvLayer,
    DownConvBlock,
    Temporal_Aggregator,
)

class UNet3PlusTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
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
        cat_channels=64, 
    ):
        super(UNet3PlusTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.pad_value = pad_value
        self.encoder = encoder
        self.cat_channels = cat_channels
        
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
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # --- DECODER (UNet 3+ Full Scale) ---
        self.up_blocks = nn.ModuleList()
        
        # FIX: Loop from (n_stages - 2) down to 0. 
        # Example: 4 stages (0,1,2,3). We need decoders for 2, 1, 0.
        # Stage 3 is the bottleneck, it does not get a decoder block.
        for i in range(self.n_stages - 2, -1, -1):
            
            # 1. Inputs from finer scales (Requires MaxPool)
            down_ops = nn.ModuleList()
            for j in range(i):
                scale_factor = 2 ** (i - j)
                down_ops.append(
                    nn.Sequential(
                        nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor, ceil_mode=True),
                        nn.Conv2d(encoder_widths[j], cat_channels, 3, padding=1),
                        nn.BatchNorm2d(cat_channels),
                        nn.ReLU(inplace=True)
                    )
                )

            # 2. Input from same scale (Direct)
            same_op = nn.Sequential(
                nn.Conv2d(encoder_widths[i], cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )

            # 3. Inputs from coarser scales (Requires Upsample)
            # FIX: Logic for determining input channels for the upsampling
            if i == self.n_stages - 2:
                # The deepest decoder block receives input directly from the Bottleneck (LTAE)
                prev_channels = encoder_widths[-1] 
            else:
                # Shallower decoder blocks receive input from the Previous Decoder Block
                # Previous block output = (num_scales) * cat_channels
                prev_channels = len(encoder_widths) * cat_channels

            up_op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(prev_channels, cat_channels, 3, padding=1),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            )

            # Total input channels to the fusion convolution
            # Finer encoders (i) + Same encoder (1) + Coarser decoder (1)
            total_in_channels = (i + 1 + 1) * cat_channels
            
            # The output of this block (standardized size)
            block_out_channels = len(encoder_widths) * cat_channels

            block = UNet3PlusBlock(
                down_ops=down_ops,
                same_op=same_op,
                up_op=up_op,
                in_channels=total_in_channels,
                out_channels=block_out_channels,
                norm="batch"
            )
            # We insert at 0, so self.up_blocks[0] corresponds to Stage 0 (Resolution 0)
            self.up_blocks.insert(0, block) 

        # --- OUT HEAD ---
        final_dim = len(encoder_widths) * cat_channels
        self.out_conv = ConvBlock(
            nkernels=[final_dim] + out_conv, 
            padding_mode=padding_mode
        )

    def forward(self, input, batch_positions=None, return_att=False):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        ) 
        
        # --- SPATIAL ENCODER ---
        out = self.in_conv.smart_forward(input)
        encoder_features = [out] # E0
        
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(encoder_features[-1])
            encoder_features.append(out) # E1 ... E_last
        
        # --- TEMPORAL BOTTLENECK ---
        bottleneck_out, att = self.temporal_encoder(
            encoder_features[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )

        # --- SPATIAL DECODER ---
        if self.return_maps:
            maps = [bottleneck_out]

        current_decoder_out = bottleneck_out
        
        # FIX: Iterate from Deepest Decoder (n_stages-2) up to Shallowest (0)
        # self.up_blocks is indexed [Stage0, Stage1, Stage2...]
        loop_indices = list(range(self.n_stages - 1)) # e.g., [0, 1, 2]
        loop_indices.reverse() # e.g., [2, 1, 0]
        
        for i in loop_indices:
            block = self.up_blocks[i]
            
            # Finer scales: 0 to i-1
            finer_encs = encoder_features[:i]
            
            # Same scale: i
            same_enc = encoder_features[i]
            
            current_decoder_out = block(
                finer_encs=finer_encs,
                same_enc=same_enc,
                prev_decoder=current_decoder_out,
                temporal_aggregator=self.temporal_aggregator,
                att_mask=att,
                pad_mask=pad_mask
            )
            
            if self.return_maps:
                maps.append(current_decoder_out)

        # --- OUTPUT ---
        if self.encoder:
             return current_decoder_out, maps
        
        out = self.out_conv(current_decoder_out)
        
        if return_att:
            return out, att
        if self.return_maps:
            return out, maps
        return out


class UNet3PlusBlock(nn.Module):
    """
    A single scale aggregation block for UNet 3+.
    It handles:
    1. Downsampling and processing finer encoder scales.
    2. Processing the same encoder scale.
    3. Upsampling the coarser decoder scale.
    4. Temporally aggregating ALL encoder inputs using the mask from the bottleneck.
    5. Concatenating and fusing.
    """
    def __init__(self, down_ops, same_op, up_op, in_channels, out_channels, norm="batch"):
        super(UNet3PlusBlock, self).__init__()
        self.down_ops = down_ops # List of operations for inputs E_0 ... E_{i-1}
        self.same_op = same_op   # Operation for E_i
        self.up_op = up_op       # Operation for D_{i+1}
        
        self.fusion = ConvLayer(
            nkernels=[in_channels, out_channels],
            norm=norm,
            k=3, s=1, p=1
        )

    def forward(self, finer_encs, same_enc, prev_decoder, temporal_aggregator, att_mask, pad_mask):
        """
        finer_encs: List of (B, T, C, H_k, W_k)
        same_enc: (B, T, C, H, W)
        prev_decoder: (B, C, H_prev, W_prev) - Already time-collapsed
        """
        
        feature_list = []
        
        # 1. Process Finer Scales (Downsample -> Conv -> Time Aggregate)
        # We must reshape B,T into B*T for the Conv2d ops, then back for aggregation
        for idx, feat in enumerate(finer_encs):
            b, t, c, h, w = feat.shape
            
            # Merge dims for spatial operation (Maxpool + Conv)
            x = feat.view(b * t, c, h, w)
            x = self.down_ops[idx](x) 
            
            # Reshape back to apply temporal attention
            _, c_new, h_new, w_new = x.shape
            x = x.view(b, t, c_new, h_new, w_new)
            
            # Aggregate Time
            x = temporal_aggregator(x, pad_mask=pad_mask, attn_mask=att_mask)
            feature_list.append(x)

        # 2. Process Same Scale (Conv -> Time Aggregate)
        b, t, c, h, w = same_enc.shape
        x = same_enc.view(b * t, c, h, w)
        x = self.same_op(x)
        _, c_new, h_new, w_new = x.shape
        x = x.view(b, t, c_new, h_new, w_new)
        
        x = temporal_aggregator(x, pad_mask=pad_mask, attn_mask=att_mask)
        feature_list.append(x)

        # 3. Process Coarser Scale (Upsample -> Conv)
        # This is already time-collapsed (it comes from the bottleneck or deeper decoder)
        x = self.up_op(prev_decoder)
        feature_list.append(x)
        
        # 4. Concatenate
        out = torch.cat(feature_list, dim=1)
        
        # 5. Fuse
        out = self.fusion(out)
        
        return out