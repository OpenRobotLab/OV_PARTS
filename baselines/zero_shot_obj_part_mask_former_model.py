# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple
import os
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
import numpy as np
import PIL.Image as Image
from detectron2.utils.file_io import PathManager
# from utils.visualizer import ColorMode, Visualizer

from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    MaskFormerObjPartClipAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer

@META_ARCH_REGISTRY.register()
class ZeroShotObjPartMaskFormer(MaskFormer):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float]
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight
    
    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    def forward(self, batched_inputs):
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        metadata = MetadataCatalog.get(dataset_name[0])
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]

        ori_images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in ori_images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        obj_instances = [x["instances"].to(self.device) for x in batched_inputs]
        ori_obj_mask = [ins.gt_masks for ins in obj_instances]

        obj_class = metadata.obj_classes[obj_instances[0].gt_classes[0].item()]
        obj_instances = self.prepare_targets(obj_instances, images)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, obj_instances)

        class_names = self.get_class_name_list(dataset_name)
        
        text_features = self.clip_adapter.get_text_features(class_names)
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )

        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["obj_part_instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            
            losses = self.criterion(outputs, targets)
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)

                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, dataset_name, ori_obj_mask[0], obj_class, text_features
                )

                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                output = sem_seg_postprocess(r, image_size, height, width)
                
                processed_results.append({"sem_seg": output})
                
                ###################  for visualization  ################
                # outfile = os.path.basename(batched_inputs[0]["file_name"]).replace('.jpg','.png')
                # image = np.array(Image.open(batched_inputs[0]["file_name"]))#[:, :, ::-1]
                # visualizer = Visualizer(image, MetadataCatalog.get('ade20k_instance_val'))
                # obj_mask = batched_inputs[0]['instances'][0].gt_masks[0]
                
                # output = output.argmax(dim=0).cpu()
                # obj_mask = F.interpolate(obj_mask.float().unsqueeze(0).unsqueeze(0), size=output.shape[-2:], mode='nearest').squeeze()
                # output[obj_mask==0.0] = 65535
                # vis_output = visualizer.draw_sem_seg(
                #         output
                #     )
                # PathManager.mkdirs(os.path.join('figs/zsseg+_zero_shot_r50_coop_voc',f'{obj_class}'))
                # vis_output.save(f'figs/zsseg+_zero_shot_r50_coop_voc/{obj_class}/{outfile}')
                # exit()

            return processed_results
    
    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes, #torch.zeros_like(targets_per_image.gt_classes),
                    "masks": padded_masks,
                }
            )
        return new_targets
    
    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        if "learnable" not in cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER:
            clip_adapter = MaskFormerClipAdapter(
                                cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
                                prompt_learner,
                                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                                mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
                                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
                            )
        else:
            clip_adapter = MaskFormerObjPartClipAdapter(cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
                                prompt_learner,
                                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                                mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
                                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,)
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT
        return init_kwargs
    
    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name, obj_mask, obj_class, text_features):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        pred = semseg.argmax(dim=0) #[h,w]
        part_masks = torch.zeros_like(semseg)
        
        for cid in torch.unique(pred):
            pmask = (pred == cid).float()
            part_masks[cid] = pmask

        if self.clip_ensemble:
            select_mask = [i for i, name in enumerate(class_names) if obj_class not in name] + [len(class_names)]
            image_features, valid_flag = self.clip_adapter(
                image, class_names, normalize=True, part_mask=part_masks, obj_mask=obj_mask, return_img=True #
            )
            if text_features is None or image_features is None:
                clip_cls = None
            else:
                clip_cls = self.clip_adapter.get_sim_logits(text_features, image_features)
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # softmax before index or after?
            else:
                clip_cls[:, select_mask] = float('-inf')
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            map_back_clip_cls = mask_cls.new_zeros(part_masks.size(0),clip_cls.size(-1))
            map_back_clip_cls[valid_flag] = clip_cls
            map_back_clip_cls = torch.einsum("mn,mhw->nhw",map_back_clip_cls, part_masks)
            semseg_clip = map_back_clip_cls

            if self.clip_ensemble_weight > 0:
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
        
                sem_seg_ensemble = trained_mask * torch.pow(
                    semseg.reshape(mask_cls.size(-1),-1).permute(1,0), self.clip_ensemble_weight
                ) * torch.pow(semseg_clip.reshape(mask_cls.size(-1),-1).permute(1,0), 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    semseg.reshape(mask_cls.size(-1),-1).permute(1,0), 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    semseg_clip.reshape(mask_cls.size(-1),-1).permute(1,0), self.clip_ensemble_weight
                )
                semseg_clip = sem_seg_ensemble.permute(1,0).reshape(-1, semseg.size(1), semseg.size(2))
            semseg = semseg_clip
        
        return semseg