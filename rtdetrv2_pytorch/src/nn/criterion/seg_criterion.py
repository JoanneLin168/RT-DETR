"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision

from ...misc import box_ops
from ...misc import dist_utils
from ...core import register

from .det_criterion import DetCriterion

@register()
class SegCriterion(DetCriterion):
    __shared__ = ['num_classes', 'use_focal_loss', 'num_sample_points']
    __inject__ = ['matcher']

    def __init__(self, 
                losses, 
                weight_dict, 
                num_classes=80, 
                alpha=0.75, 
                gamma=2.0, 
                box_fmt='cxcywh',
                matcher=None,
                
                num_sample_points=12544,
                oversample_ratio=3.0,
                important_sample_ratio=0.75):
        """
        Args:
            losses (list[str]): requested losses, support ['boxes', 'vfl', 'focal']
            weight_dict (dict[str, float)]: corresponding losses weight, including
                ['loss_bbox', 'loss_giou', 'loss_vfl', 'loss_focal']
            box_fmt (str): in box format, 'cxcywh' or 'xyxy'
            matcher (Matcher): matcher used to match source to target
        """
        super().__init__(losses, weight_dict, num_classes, alpha, gamma, box_fmt, matcher)
        self.losses = losses
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.box_fmt = box_fmt
        assert matcher is not None, ''
        self.matcher = matcher

        assert oversample_ratio >= 1
        assert important_sample_ratio <= 1 and important_sample_ratio >= 0

        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.important_sample_ratio = important_sample_ratio
        self.num_oversample_points = int(num_sample_points * oversample_ratio)
        self.num_important_points = int(num_sample_points *
                                        important_sample_ratio)
        self.num_random_points = num_sample_points - self.num_important_points

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_masks=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(SegCriterion, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            masks=masks,
            gt_mask=gt_mask,
            num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = DetCriterion.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(SegCriterion, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                masks=dn_out_masks,
                gt_mask=gt_mask,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': torch.Tensor([0.], device=logits.device)
                 for k in total_loss.keys()})

        return total_loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.Tensor([0.], device=masks.device)
            loss[name_dice] = torch.Tensor([0.], device=masks.device)
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        # sample points
        sample_points = self._get_point_coords_by_uncertainty(src_masks)
        sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0

        src_masks = F.grid_sample(
            src_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2])

        target_masks = F.grid_sample(
            target_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2]).detach()

        loss[name_mask] = self.loss_coeff[
            'mask'] * F.binary_cross_entropy_with_logits(
                src_masks, target_masks,
                reduction='none').mean(1).sum() / num_gts
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _get_point_coords_by_uncertainty(self, masks):
        # Sample points based on their uncertainty.
        masks = masks.detach()
        num_masks = masks.shape[0]
        sample_points = torch.rand(
            [num_masks, 1, self.num_oversample_points, 2])

        out_mask = F.grid_sample(
            masks.unsqueeze(1), 2.0 * sample_points - 1.0,
            align_corners=False).squeeze([1, 2])
        out_mask = -torch.abs(out_mask)

        _, topk_ind = torch.topk(out_mask, self.num_important_points, axis=1)
        batch_ind = torch.arange(end=num_masks, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_important_points])
        topk_ind = torch.stack([batch_ind, topk_ind], axis=-1)

        sample_points = torch.gather_nd(sample_points.squeeze(1), topk_ind)
        if self.num_random_points > 0:
            sample_points = torch.concat(
                [
                    sample_points,
                    torch.rand([num_masks, self.num_random_points, 2])
                ],
                axis=1)
        return sample_points