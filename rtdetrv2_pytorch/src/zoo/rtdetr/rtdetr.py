"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register

from .segmentation import *


__all__ = ['RTDETR', 'RTDETRSegm', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


@register()
class RTDETRSegm(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module,
        freeze_detr: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

        # TODO: figure out how to freeze the DETR model
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = self.decoder.hidden_dim, self.decoder.nhead
        # self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        # self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
        
    # REFERENCE: https://github.com/facebookresearch/detr/blob/main/models/segmentation.py#L37
    def forward(self, x, targets=None):
        # features = self.backbone(x)
        # bs = x.shape[0]

        # # Get attention mask using the last feature map
        # src = features[-1]
        # n, _, h, w = src.shape
        # mask = torch.ones((n, h, w), dtype=torch.bool, device=src.device)
        # print("src", src.shape)

        # enc_feats = self.encoder(features)
        # out, hs, memory = self.decoder(enc_feats, targets)
        # print("hs", hs.shape)

        # # FIXME h_boxes takes the last one computed, keep this in mind
        # src_proj = enc_feats[-1]
        # bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        # seg_masks = self.mask_head(src_proj, bbox_mask, [features[2], features[1], features[0]])
        # outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        # out["pred_masks"] = outputs_seg_masks

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 