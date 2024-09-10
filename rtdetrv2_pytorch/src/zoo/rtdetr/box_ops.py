"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def mask_to_box_coordinate(masks, normalize=False, format="xyxy", dtype=torch.float):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    assert len(masks) == 4

    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    out_bbox = torch.stack([x_min, y_min, x_max, y_max], 1)

    if normalize:
        out_bbox /= torch.tensor([w, h, w, h], dtype=dtype, device=out_bbox.device)

    return out_bbox if format == "xyxy" else box_xyxy_to_cxcywh(out_bbox)

# def mask_to_box_coordinate(mask,
#                            normalize=False,
#                            format="xyxy",
#                            dtype="float32"):
#     """
#     Compute the bounding boxes around the provided mask.
#     Args:
#         mask (Tensor:bool): [b, c, h, w]

#     Returns:
#         bbox (Tensor): [b, c, 4]
#     """
#     assert mask.ndim == 4
#     assert format in ["xyxy", "xywh"]

#     h, w = mask.shape[-2:]
#     y, x = torch.meshgrid(
#         torch.arange(
#             end=h, dtype=dtype), torch.arange(
#                 end=w, dtype=dtype))

#     x_mask = x * mask.astype(x.dtype)
#     x_max = x_mask.flatten(-2).max(-1) + 1
#     x_min = torch.where(mask.astype(bool), x_mask,
#                          torch.to_tensor(1e8)).flatten(-2).min(-1)

#     y_mask = y * mask.astype(y.dtype)
#     y_max = y_mask.flatten(-2).max(-1) + 1
#     y_min = torch.where(mask.astype(bool), y_mask,
#                          torch.to_tensor(1e8)).flatten(-2).min(-1)
#     out_bbox = torch.stack([x_min, y_min, x_max, y_max], axis=-1)
#     mask = mask.any(axis=[2, 3]).unsqueeze(2)
#     out_bbox = out_bbox * mask.astype(out_bbox.dtype)
#     if normalize:
#         out_bbox /= torch.to_tensor([w, h, w, h]).astype(dtype)

#     return out_bbox if format == "xyxy" else box_xyxy_to_cxcywh(out_bbox)