"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .box_ops import mask_to_box_coordinate


from ...core import register


__all__ = ['MaskRTDETRTransformer']

from .rtdetr_decoder import TransformerDecoderLayer, MLP


def _get_pred_mask(out_query,
                    mask_feat,
                    mask_query_head):
    mask_query_embed = mask_query_head(out_query)
    batch_size, mask_dim, _ = mask_query_embed.shape
    _, _, mask_h, mask_w = mask_feat.shape
    out_mask = torch.bmm(
        mask_query_embed, mask_feat.flatten(2)).reshape(
        [batch_size, mask_dim, mask_h, mask_w])
    return out_mask



class MaskTransformerDecoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 eval_idx=-1,
                 eval_topk=100):
        super(MaskTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.eval_topk = eval_topk

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                mask_feat,
                bbox_head,
                score_head,
                query_pos_head,
                mask_query_head,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_masks = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(inverse_sigmoid(ref_points_detach))

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                logits_ = score_head[i](output)
                masks_ = _get_pred_mask(
                    output, mask_feat, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                logits_ = score_head[i](output)
                masks_ = _get_pred_mask(
                    output, mask_feat, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                dec_out_bboxes.append(inter_ref_bbox)
                return (torch.stack(dec_out_bboxes),
                        torch.stack(dec_out_logits),
                        torch.stack(dec_out_masks))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return (torch.stack(dec_out_bboxes),
                torch.stack(dec_out_logits),
                torch.stack(dec_out_masks))


@register()
class MaskRTDETRTransformer(nn.Module):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_spatial_size', 'num_prototypes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_prototypes=32,
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.4,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 query_pos_head_inv_sig=False,
                 mask_enhanced=True,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 version='v1'):
        
        super(MaskRTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.mask_enhanced = mask_enhanced
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_points)
        self.decoder = MaskTransformerDecoder(hidden_dim, decoder_layer,
                                              num_layers, eval_idx)

        # denoising part
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim,
                                  hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # mask embedding
        self.mask_query_head = MLP(hidden_dim, hidden_dim,
                                   num_prototypes, num_layers=3)

        # encoder head
        if version == 'v1':
            self.enc_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim,)
            )
        else:
            self.enc_output = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(hidden_dim, hidden_dim)),
                ('norm', nn.LayerNorm(hidden_dim,)),
            ]))
        
        # shared prediction head
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'feat_channels': [i.channels for i in input_shape],
                'feat_strides': [i.stride for i in input_shape]}

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    """
    TODO: implement the following
    / figure out how to get mask_feat (from maskhybridencoder)
    - fix your _get_decodder_input(); currently its a mess of paddle maskrtdetr, rtdetr, and rtdetrv2
    - rn you have done some simplified outputs to get things working; fix this
    """
    def _get_decoder_input(self,
                           memory,
                           mask_feat,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask

        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 
        output_memory = self.enc_output(memory) # this has norm layer, i.e. dec_norm

        enc_out_logits = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors


        # get topk index
        _, topk_ind = torch.topk(enc_out_logits.max(-1).values, self.num_queries, dim=1)

        # extract content and position query embedding
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        
        # get encoder output: {logits, bboxes, masks}
        # (since output_memory is dec_norm(query_embed), don't need dec_norm)
        enc_out_masks = _get_pred_mask(
            target, mask_feat, self.mask_query_head)
        enc_out_bboxes = F.sigmoid(reference_points_unact)
        enc_out = (enc_out_logits, enc_out_bboxes, enc_out_masks)

        # concat denoising query
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = target.detach()
        
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        # TODO: line 402 (torch.concat) isn't concatenating them: [torch.Size([4, 198, 4]),torch.Size([4, 4]) ]
            # likely you aren't supposed to get denoising_bbox_unact so early
        # if self.mask_enhanced:
        #     # use mask-enhanced anchor box initialization
        #     reference_points = mask_to_box_coordinate(
        #         enc_out_masks > 0, normalize=True, format="xywh")
        #     reference_points_unact = inverse_sigmoid(reference_points)
        #     print(reference_points_unact.shape, reference_points.shape)
        
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)

        # direct prediction from the matching and denoising part in the beginning
        if self.training and denoising_class is not None:
            init_out_logits = self.enc_score_head(target)
            init_out_masks = _get_pred_mask(
                target, mask_feat, self.mask_query_head)
            init_out_bboxes = F.sigmoid(reference_points_unact)
            init_out = (init_out_logits, init_out_bboxes, init_out_masks)
        else:
            init_out = None

        return target, reference_points_unact.detach(), enc_out, init_out
    
    
    def forward(self, enc_feats, mask_feat, targets=None, pad_mask=None):
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(enc_feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_out, init_out = \
            self._get_decoder_input(memory, mask_feat, spatial_shapes, denoising_class, denoising_bbox_unact)
        
        enc_topk_logits, enc_topk_bboxes, enc_topk_masks = enc_out

        # decoder
        out_bboxes, out_logits, out_masks = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            mask_feat,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.mask_query_head,
            attn_mask=attn_mask,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig)
        

        # DETR head + mask
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_masks, out_masks = torch.split(out_masks, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_masks': out_masks[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_masks[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes], [enc_topk_masks]))
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_out_masks)
                out['dn_meta'] = dn_meta


        # # MaskDINO Head

        # # TODO: make this into a separate class
        # dec_out_logits = out_logits
        # dec_out_bboxes = out_bboxes
        # dec_out_masks = out_masks

        # if self.training and dn_meta is not None:
        #     dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
        #     dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
        #     dn_out_masks, out_masks = torch.split(out_masks, dn_meta['dn_num_split'], dim=2)

        #     if init_out is not None:
        #         init_out_logits, init_out_bboxes, init_out_masks = init_out

        #         init_out_logits_dn, init_out_logits = torch.split(
        #             init_out_logits, dn_meta['dn_num_split'], dim=1)
        #         init_out_bboxes_dn, init_out_bboxes = torch.split(
        #             init_out_bboxes, dn_meta['dn_num_split'], dim=1)
        #         init_out_masks_dn, init_out_masks = torch.split(
        #             init_out_masks, dn_meta['dn_num_split'], dim=1)

        #         dec_out_logits = torch.concat(
        #             [init_out_logits.unsqueeze(0), dec_out_logits])
        #         dec_out_bboxes = torch.concat(
        #             [init_out_bboxes.unsqueeze(0), dec_out_bboxes])
        #         dec_out_masks = torch.concat(
        #             [init_out_masks.unsqueeze(0), dec_out_masks])

        #         dn_out_logits = torch.concat(
        #             [init_out_logits_dn.unsqueeze(0), dn_out_logits])
        #         dn_out_bboxes = torch.concat(
        #             [init_out_bboxes_dn.unsqueeze(0), dn_out_bboxes])
        #         dn_out_masks = torch.concat(
        #             [init_out_masks_dn.unsqueeze(0), dn_out_masks])
        # else:
        #     dn_out_bboxes, dn_out_logits = None, None
        #     dn_out_masks = None


        # enc_out_logits, enc_out_bboxes, enc_out_masks = enc_out
        # out_logits = torch.concat(
        #     [enc_out_logits.unsqueeze(0), dec_out_logits])
        # out_bboxes = torch.concat(
        #     [enc_out_bboxes.unsqueeze(0), dec_out_bboxes])
        # out_masks = torch.concat(
        #     [enc_out_masks.unsqueeze(0), dec_out_masks])


        # out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_masks': out_masks[-1]}

        # if self.training and self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_masks[:-1])
        #     out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes], [enc_topk_masks]))
            
        #     if self.training and dn_meta is not None:
        #         out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_out_masks)
        #         out['dn_meta'] = dn_meta

        return out

        # return out_logits, out_bboxes, out_masks, enc_out, init_out, dn_meta
    
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                for a, b, c in zip(outputs_class, outputs_coord, outputs_mask)]