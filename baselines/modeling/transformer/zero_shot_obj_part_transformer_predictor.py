# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from mmseg.ops import resize
from .transformer_predictor import TransformerPredictor, MLP
from einops import rearrange, repeat

from timm.models.layers import Mlp, DropPath, to_2tuple


class ZeroShotTransformerObjPartPredictor(TransformerPredictor):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        embedding_dim: int,
        embed_hidden_dim: int,
        embed_layers: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        super().__init__(
            in_channels,
            False,
            num_classes=embedding_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            deep_supervision=deep_supervision,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
        )
        self.mask_classification = mask_classification
        # output FFNs
        if self.mask_classification:
            self.class_embed = MLP(
                hidden_dim, embed_hidden_dim, embedding_dim, embed_layers
            )
            # self.class_embed = nn.Linear(hidden_dim, 2)
    

    def freeze_pretrained(self):
        for name, module in self.named_children():
            if name not in ["class_embed"]:
                for param in module.parameters():
                    param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["embedding_dim"] = cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM
        ret["embed_hidden_dim"] = cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM
        ret["embed_layers"] = cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, obj_instances):
        obj_masks = torch.stack([obj['masks'].float() for obj in obj_instances])
        pos = self.pe_layer(x)

        src = x
        mask = None
        mask_features_resize = resize(input=mask_features, size=obj_masks.shape[2:]) * obj_masks
        obj_mask_features = nn.AvgPool2d(kernel_size=mask_features_resize.shape[2:])(mask_features_resize).squeeze(-1).squeeze(-1)     
        
        obj_masks = F.max_pool2d(obj_masks, 32, 32)

        mask = 1.0 - obj_masks
        hs, memory = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos
        )

        if self.mask_classification:
            outputs_class = self.class_embed(hs + obj_mask_features.unsqueeze(0).unsqueeze(2))
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}
        
        

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum(
                "lbqc,bchw->lbqhw", mask_embed, mask_features
            )
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum(
                "bqc,bchw->bqhw", mask_embed, mask_features
            )
            out["pred_masks"] = outputs_seg_masks
        return out
        
        