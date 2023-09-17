# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.zero_shot_obj_part_transformer_predictor import ZeroShotTransformerObjPartPredictor
from .pixel_decoder import build_pixel_decoder
from .zero_shot_mask_former_head import ZeroShotMaskFormerHead

@SEM_SEG_HEADS_REGISTRY.register()
class ZeroShotObjPartMaskFormerHead(ZeroShotMaskFormerHead):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            pixel_decoder=pixel_decoder,
            loss_weight=loss_weight,
            ignore_value=ignore_value,
            transformer_predictor=transformer_predictor,
            transformer_in_feature=transformer_in_feature
        )
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": ZeroShotTransformerObjPartPredictor(
                cfg,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder"
                else input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels,
                mask_classification=True,
            ),
        }
    
    def forward(self, features, obj_masks):
        return self.layers(features, obj_masks)
    
    def layers(self, features, obj_masks):
        (
            mask_features,
            transformer_encoder_features,
        ) = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features, obj_masks)
        else:
            predictions = self.predictor(
                features[self.transformer_in_feature], mask_features, obj_masks
            )
        return predictions