# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from .. import builder
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector

import time

@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.t_voxelize_all = []
        self.t_voxel_encoder_all = []
        self.t_middle_encoder_all = []
        self.t_backbone_all = []
        self.t_neck_all = []
        self.t_bbox3d2result_all = []
        self.t_extract_feat_all = []
        self.t_bbox_head_all = []
        self.t_get_bboxes_all = []

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        torch.cuda.synchronize()
        t0 = time.time()
        voxels, num_points, coors = self.voxelize(points)
        torch.cuda.synchronize()
        t1 = time.time()
        # try:
        #     voxels, num_points, coors = self.voxelize(points)
        # except RuntimeError:
        #     print(img_metas)

        # print(voxels, num_points, coors) 
        # voxels, num_points, coors = self.voxelize(points)
        # voxel_features = self.voxel_encoder(voxels, num_points, coors)
        voxel_features = self.voxel_encoder(voxels, coors)
        torch.cuda.synchronize()
        t2 = time.time()
        if len(coors) == 0:
            batch_size = 6
        else:
            batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        torch.cuda.synchronize()
        t3 = time.time()
        x = self.backbone(x)
        torch.cuda.synchronize()
        t4 = time.time()
        if self.with_neck:
            x = self.neck(x)
        torch.cuda.synchronize()
        t5 = time.time()

        t_voxelize = (t1-t0)*1000
        t_voxel_encoder = (t2-t1)*1000
        t_middle_encoder = (t3-t2)*1000
        t_backbone = (t4-t3)*1000
        t_neck = (t5-t4)*1000
        print(' ')
        print(' t_voxelize: ', t_voxelize)
        print(' t_voxel_encoder: ', t_voxel_encoder)
        print(' t_middle_encoder: ', t_middle_encoder)
        print(' t_backbone: ', t_backbone)
        print(' t_neck: ', t_neck)
        self.t_voxelize_all.append(t_voxelize)
        print('t_voxelize_mean: ', sum(self.t_voxelize_all)/len(self.t_voxelize_all))
        self.t_voxel_encoder_all.append(t_voxel_encoder)
        print('t_voxel_encoder_mean: ', sum(self.t_voxel_encoder_all)/len(self.t_voxel_encoder_all))
        self.t_middle_encoder_all.append(t_middle_encoder)
        print('t_middle_encoder_mean: ', sum(self.t_middle_encoder_all)/len(self.t_middle_encoder_all))
        self.t_backbone_all.append(t_backbone)
        print('t_backbone_mean: ', sum(self.t_backbone_all)/len(self.t_backbone_all))
        self.t_neck_all.append(t_neck)
        print('t_neck_mean: ', sum(self.t_neck_all)/len(self.t_neck_all))
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        torch.cuda.synchronize()
        t0 = time.time()
        x = self.extract_feat(points, img_metas)

        torch.cuda.synchronize()
        t1 = time.time()
        outs = self.bbox_head(x)

        torch.cuda.synchronize()
        t2 = time.time()
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        torch.cuda.synchronize()
        t3 = time.time()
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        torch.cuda.synchronize()
        t4 = time.time()

        t_extract_feat = (t1-t0)*1000
        t_bbox_head = (t2-t1)*1000
        t_get_bboxes = (t3-t2)*1000
        t_bbox3d2result = (t4-t3)*1000
        self.t_extract_feat_all.append(t_extract_feat)
        self.t_bbox_head_all.append(t_bbox_head)
        self.t_get_bboxes_all.append(t_get_bboxes)
        self.t_bbox3d2result_all.append(t_bbox3d2result)
        try:
            print('t_extract_feat_mean: ', sum(self.t_extract_feat_all[len(self.t_extract_feat_all)//2:])/(len(self.t_extract_feat_all)//2))
            print('t_bbox_head_mean: ', sum(self.t_bbox_head_all[len(self.t_bbox_head_all)//2:])/(len(self.t_bbox_head_all)//2))
            print('t_get_bboxes_mean: ', sum(self.t_get_bboxes_all[len(self.t_get_bboxes_all)//2:])/(len(self.t_get_bboxes_all)//2))
            print('t_bbox3d2result_mean: ', sum(self.t_bbox3d2result_all[len(self.t_bbox3d2result_all)//2:])/(len(self.t_bbox3d2result_all)//2))
        except ZeroDivisionError:
            print(' ')
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
