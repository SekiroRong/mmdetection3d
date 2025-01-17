import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from ..builder import BACKBONES
import time
import MinkowskiEngine as ME

t_bb_all = []
@BACKBONES.register_module()
class SparseSECOND(BaseModule):
    """Backbone network for SparsePointPillars.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SparseSECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                ME.MinkowskiConvolution(in_filters[i],
                                        out_channels[i],
                                        2,
                                        bias=False,
                                        stride=layer_strides[i],
                                        dimension=2,
                                        expand_coordinates=True),
                ME.MinkowskiBatchNorm(out_channels[i], eps=1e-3, momentum=0.01),
                ME.MinkowskiReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    ME.MinkowskiConvolution(out_channels[i],
                                            out_channels[i],
                                            3,
                                            bias=False,
                                            dimension=2))
                block.append(
                    ME.MinkowskiBatchNorm(out_channels[i],
                                          eps=1e-3,
                                          momentum=0.01))
                block.append(ME.MinkowskiReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.layer_strides = layer_strides

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        input_shape = x.shape
        vals = x._values()
        idxs = x._indices().permute(1, 0).contiguous().int()
        x = ME.SparseTensor(vals, idxs)
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return input_shape, self.layer_strides, tuple(outs)