"""
Taken from https://github.com/russel0719/UNet-3-Plus-Pytorch/blob/main/models/unet3plus.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply 3*3 convolutions with BN and ReLU.
    """
    layers = []
    for i in range(1, n + 1):
        conv = nn.Conv2d(in_channels=in_channels if i == 1 else out_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding if padding != 'same' else 'same',
                         bias=not is_bn)  # Disable bias when using BatchNorm
        layers.append(conv)
        
        if is_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def dot_product(seg, cls):
    b, n, h, w = seg.shape
    seg = seg.view(b, n, -1)
    cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
    final = torch.einsum("bik,bi->bik", seg, cls)
    final = final.view(b, n, h, w)
    return final

class UNet3Plus(nn.Module):
    def __init__(self, input_shape, output_channels, deep_supervision=False, cgm=False, training=False):
        super(UNet3Plus, self).__init__()
        self.deep_supervision = deep_supervision
        self.CGM = deep_supervision and cgm
        self.training = training

        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]
        self.cat_blocks = len(self.filters)
        self.upsample_channels = self.cat_blocks * self.cat_channels

        # Encoder
        self.e1 = ConvBlock(input_shape[0], self.filters[0])
        self.e2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[0], self.filters[1])
        )
        self.e3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[1], self.filters[2])
        )
        self.e4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[2], self.filters[3])
        )
        self.e5 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[3], self.filters[4])
        )

        # Classification Guided Module
        self.cgm = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.filters[4], 2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        ) if self.CGM else None

        # Decoder
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.filters[3], self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d3 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d3_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d2 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d2_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d1 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d1_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.final = nn.Conv2d(self.upsample_channels, output_channels, kernel_size=1)

        # Deep Supervision
        self.deep_sup = nn.ModuleList([
                ConvBlock(self.upsample_channels, output_channels, n=1, is_bn=False, is_relu=False)
                for _ in range(3)
            ] + [ConvBlock(self.filters[4], output_channels, n=1, is_bn=False, is_relu=False)]
        ) if self.deep_supervision else None

    def forward(self, x, batch_positions=None, **kwargs) -> torch.Tensor:
        """
        Support both 4-D inputs (B, C, H, W) and 5-D temporal inputs (B, T, C, H, W).
        This implements a small "smart forward" behaviour: modules that are 2D
        (conv blocks) are applied per-frame when the input is 5-D by flattening
        the (B, T) dims before applying them, while functional ops (pool/interp)
        are also applied per-frame. Concatenation across channels is handled
        for both cases.
        """
        training = self.training
        is_temporal = x.dim() == 5
        if is_temporal:
            b, t, _, _, _ = x.shape

        def apply_module(mod, inp):
            # Apply a nn.Module that expects 4-D input to either 4-D or 5-D tensors.
            if inp.dim() == 4:
                return mod(inp)
            b, t, c, h, w = inp.shape
            out = mod(inp.view(b * t, c, h, w))
            _, oc, oh, ow = out.shape
            return out.view(b, t, oc, oh, ow)

        def apply_fn(fn, inp, *args, **kwargs):
            # Apply a functional that expects 4-D input (like F.max_pool2d / F.interpolate)
            if inp.dim() == 4:
                return fn(inp, *args, **kwargs)
            b, t, c, h, w = inp.shape
            out = fn(inp.view(b * t, c, h, w), *args, **kwargs)
            # out is expected 4-D
            _, oc, oh, ow = out.shape
            return out.view(b, t, oc, oh, ow)

        def cat_channels(tensors):
            # Concatenate along channel dimension for 4-D and 5-D tensors
            if tensors[0].dim() == 4:
                return torch.cat(tensors, dim=1)
            else:
                return torch.cat(tensors, dim=2)
        # Encoder (use smart apply so modules work with 4-D or 5-D inputs)
        e1 = apply_module(self.e1, x)
        e2 = apply_module(self.e2, e1)
        e3 = apply_module(self.e3, e2)
        e4 = apply_module(self.e4, e3)
        e5 = apply_module(self.e5, e4)

        # Classification Guided Module
        if self.CGM:
            cls = self.cgm(e5)
            cls = torch.argmax(cls, dim=1).float()

        # Decoder
        d4 = [
            apply_fn(F.max_pool2d, e1, 8),
            apply_fn(F.max_pool2d, e2, 4),
            apply_fn(F.max_pool2d, e3, 2),
            e4,
            apply_fn(F.interpolate, e5, scale_factor=2, mode='bilinear', align_corners=True),
        ]
        d4 = [apply_module(conv, d) for conv, d in zip(self.d4, d4)]
        d4 = cat_channels(d4)
        d4 = apply_module(self.d4_conv, d4)

        d3 = [
            apply_fn(F.max_pool2d, e1, 4),
            apply_fn(F.max_pool2d, e2, 2),
            e3,
            apply_fn(F.interpolate, d4, scale_factor=2, mode='bilinear', align_corners=True),
            apply_fn(F.interpolate, e5, scale_factor=4, mode='bilinear', align_corners=True),
        ]
        d3 = [apply_module(conv, d) for conv, d in zip(self.d3, d3)]
        d3 = cat_channels(d3)
        d3 = apply_module(self.d3_conv, d3)

        d2 = [
            apply_fn(F.max_pool2d, e1, 2),
            e2,
            apply_fn(F.interpolate, d3, scale_factor=2, mode='bilinear', align_corners=True),
            apply_fn(F.interpolate, d4, scale_factor=4, mode='bilinear', align_corners=True),
            apply_fn(F.interpolate, e5, scale_factor=8, mode='bilinear', align_corners=True),
        ]
        d2 = [apply_module(conv, d) for conv, d in zip(self.d2, d2)]
        d2 = cat_channels(d2)
        d2 = apply_module(self.d2_conv, d2)

        d1 = [
            e1,
            apply_fn(F.interpolate, d2, scale_factor=2, mode='bilinear', align_corners=True),
            apply_fn(F.interpolate, d3, scale_factor=4, mode='bilinear', align_corners=True),
            apply_fn(F.interpolate, d4, scale_factor=8, mode='bilinear', align_corners=True),
            apply_fn(F.interpolate, e5, scale_factor=16, mode='bilinear', align_corners=True),
        ]
        d1 = [apply_module(conv, d) for conv, d in zip(self.d1, d1)]
        d1 = cat_channels(d1)
        d1 = apply_module(self.d1_conv, d1)
        d1 = apply_module(self.final, d1)

        outputs = [d1]

        # Deep Supervision
        if self.deep_supervision and training:
            outputs.extend([
                F.interpolate(self.deep_sup[0](d2), scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[1](d3), scale_factor=4, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[2](d4), scale_factor=8, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[3](e5), scale_factor=16, mode='bilinear', align_corners=True)
            ])

        # Classification Guided Module
        if self.CGM:
            outputs = [dot_product(out, cls) for out in outputs]
        
        outputs = [F.sigmoid(out) for out in outputs]

        if is_temporal:
            # reshape each output from (B*T, C, H, W) -> (B, T, C, H, W) and
            # aggregate over time (mean). This yields (B, C, H, W) per output.
            outputs = [out.view(b, t, out.shape[1], out.shape[2], out.shape[3]).mean(dim=1) for out in outputs]

        if self.deep_supervision and training:
            return torch.cat(outputs, dim=0)
        else:
            return outputs[0]

if __name__ == "__main__":
    INPUT_SHAPE = [1, 320, 320]
    OUTPUT_CHANNELS = 2

    unet_3P = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=False, cgm=False, training=True)
    unet_3P_deep_sup = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=True, cgm=False, training=True)
    unet_3P_deep_sup_cgm = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=True, cgm=True, training=True)
    # print(unet_3P)

    # Example input tensor
    x = torch.randn(4, *INPUT_SHAPE)

    # Forward pass
    output = unet_3P(x)
    print(f"Output shape: {output.shape}")
    
    output = unet_3P_deep_sup(x)
    print(f"Output shape: {output.shape}")
    
    output = unet_3P_deep_sup_cgm(x)
    print(f"Output shape: {output.shape}")