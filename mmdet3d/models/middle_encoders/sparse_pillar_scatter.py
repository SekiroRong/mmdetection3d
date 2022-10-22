import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS

import time

@MIDDLE_ENCODERS.register_module()
class SparsePointPillarsScatter(nn.Module):
    """Sparse Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image in 
    sparse tensor format.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels=64, output_shape=[504, 440]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    #@auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.
        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        sparse_coords = coors[:, [0, 2, 3]]
        # assert (out_coords[:1] < num_voxels[0]).all()
        # assert (out_coords[:2] < num_voxels[1]).all()
        # print(sparse_coords.shape)
        # print(voxel_features.shape)
        out_shape = (batch_size, self.ny, self.nx, voxel_features.shape[1])
        # print(out_shape)
        sp_batch = torch.sparse_coo_tensor(sparse_coords.t(), voxel_features,out_shape)
        # print(sp_batch.shape)
        return sp_batch