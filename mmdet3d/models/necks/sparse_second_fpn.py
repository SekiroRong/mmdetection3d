import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from ..builder import NECKS
import time
import MinkowskiEngine as ME

class ToDenseMink(nn.Module):

    def __init__(self, input_shape, first_shrink_stride, first_upsample_stride,
                 out_size):
        super(ToDenseMink, self).__init__()
        batch_size, x_size, y_size, _ = input_shape
        scale = first_shrink_stride // first_upsample_stride
        self.output_shape = torch.Size(
            [batch_size, out_size, x_size // scale, y_size // scale])
        self.min_coord = torch.zeros((2, ), dtype=torch.int)

    def forward(self, x):
        # print(x.device)
        return x.dense(shape=self.output_shape,
                       min_coordinate=self.min_coord)[0]

@NECKS.register_module()
class SparseSECONDFPN(BaseModule):
    """FPN used in SparsePointPillars.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SparseSECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = ME.MinkowskiConvolutionTranspose(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False,
                    dimension=2)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = ME.MinkowskiConvolution(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=False,
                    dimension=2)

            deblock = nn.Sequential(
                upsample_layer,
                ME.MinkowskiBatchNorm(out_channel, eps=1e-3, momentum=0.01),
                ME.MinkowskiReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.init_weights()
        self.upsample_strides = upsample_strides

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            torch.Tensor: Feature maps.
        """
        input_shape, layer_strides, x = x
        # print(input_shape, layer_strides)
        assert len(x) == len(self.in_channels)
        to_dense = ToDenseMink(input_shape, layer_strides[0],
                               self.upsample_strides[0], self.out_channels[0])
        ups = [
            to_dense(deblock(x[i])) for i, deblock in enumerate(self.deblocks)
        ]
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]