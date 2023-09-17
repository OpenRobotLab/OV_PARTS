# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os
import numpy as np
import torch
from torch.nn import functional as F
import pycocotools.mask as mask_utils
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from baselines.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from PIL import Image

__all__ = ["SemanticObjPartDatasetMapper"]

class SemanticObjPartDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        obj_map,
        obj_part_map
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.obj_map = obj_map
        self.obj_part_map = obj_part_map

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # augs = [CropImageWithMask(cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO)]
        
        if is_train:
            # Assume always applies to the training set.
            dataset_names = cfg.DATASETS.TRAIN
            meta = MetadataCatalog.get(dataset_names[0])
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        meta.ignore_label,
                    )
                )

            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())
        else:
            dataset_names = cfg.DATASETS.TEST
            meta = MetadataCatalog.get(dataset_names[0])
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
            augs = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
            
        ignore_label = meta.ignore_label
        if hasattr(meta, "obj_map"):
            obj_map = meta.obj_map
        else:
            obj_map = None
        
        if hasattr(meta, "obj_part_map"):
            obj_part_map = meta.obj_part_map
        else:
            obj_part_map = None

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if is_train else -1,
            "obj_map": obj_map,
            "obj_part_map": obj_part_map
        }
        return ret
    
    def ann_to_rle(self, ann, h, w):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.
        Args:
            ann (dict) : annotation object
        Returns:
            ann (rle)
        """

        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def ann_to_mask(self, ann, height, width):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object
        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann, height, width)
        return mask_utils.decode(rle)

    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]

        ##### Load obj part segmentation map
        if dataset_dict["sem_seg_file_name"] != 'NA':
            # PyTorch transformation not implemented for uint16, so converting it to double first、
            file_name = dataset_dict["sem_seg_file_name"]
            obj_part_sem_seg_gt = utils.read_image(file_name).astype(
                "double"
            )
        elif 'obj_part_annotations' in dataset_dict:
            obj_part_annos = dataset_dict['obj_part_annotations']
            obj_part_sem_seg_gt = np.zeros(image_shape, dtype=np.float32) + self.ignore_label
            for i,anno in enumerate(obj_part_annos):
                mask = self.ann_to_mask(anno, dataset_dict['height'], dataset_dict['width'])
                obj_part_sem_seg_gt[mask==1] = anno['category_id']
        else:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        
        ##### Load obj segmentation map
        if dataset_dict["obj_sem_seg_file_name"] != 'NA':
            # PyTorch transformation not implemented for uint16, so converting it to double first、
            file_name = dataset_dict["obj_sem_seg_file_name"]
            obj_sem_seg_gt = utils.read_image(file_name).astype(
                "double"
            )
        elif 'obj_annotations' in dataset_dict:
            obj_annos = dataset_dict['obj_annotations']
            obj_sem_seg_gt = np.zeros(image_shape, dtype=np.float32) + self.ignore_label
            for i,anno in enumerate(obj_annos):
                mask = self.ann_to_mask(anno, dataset_dict['height'], dataset_dict['width'])
                obj_sem_seg_gt[mask==1] = anno['category_id']
        else:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        dataset_dict['ori_obj_part_sem_seg_gt'] = obj_part_sem_seg_gt
        dataset_dict['ori_obj_sem_seg_gt'] = obj_sem_seg_gt
        
        if self.obj_part_map is not None:
            mapped_obj_part_sem_seg_gt = np.zeros_like(obj_part_sem_seg_gt, dtype=np.float) + self.ignore_label
            for old_id, new_id in self.obj_part_map.items():
                mapped_obj_part_sem_seg_gt[obj_part_sem_seg_gt == old_id] = new_id
            dataset_dict['ori_obj_part_sem_seg_gt'] = mapped_obj_part_sem_seg_gt
            
        aug_input = T.AugInput2(image, sem_seg=obj_sem_seg_gt, part_sem_seg=obj_part_sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        obj_sem_seg_gt = aug_input.sem_seg
        obj_part_sem_seg_gt = aug_input.part_sem_seg
        
        if "category_id" in dataset_dict: ### obj binary mask mode
            obj_sem_seg_gt[obj_sem_seg_gt != dataset_dict["category_id"]] = self.ignore_label
            obj_part_sem_seg_gt[obj_sem_seg_gt != dataset_dict["category_id"]] = self.ignore_label
        
        if self.obj_map is not None:
            mapped_obj_sem_seg_gt = np.zeros_like(obj_sem_seg_gt, dtype=np.float) + self.ignore_label
            for old_id, new_id in self.obj_map.items():
                mapped_obj_sem_seg_gt[obj_sem_seg_gt == old_id] = new_id
            obj_sem_seg_gt = mapped_obj_sem_seg_gt
        
        if self.obj_part_map is not None:
            mapped_obj_part_sem_seg_gt = np.zeros_like(obj_part_sem_seg_gt, dtype=np.float) + self.ignore_label
            for old_id, new_id in self.obj_part_map.items():
                mapped_obj_part_sem_seg_gt[obj_part_sem_seg_gt == old_id] = new_id
            obj_part_sem_seg_gt = mapped_obj_part_sem_seg_gt
            
        if len(np.unique(obj_part_sem_seg_gt)) == 1 and np.unique(obj_part_sem_seg_gt)[0] == self.ignore_label:
            return None
        
         # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if obj_sem_seg_gt is not None:
            obj_sem_seg_gt = torch.as_tensor(obj_sem_seg_gt.astype("long"))
        if obj_part_sem_seg_gt is not None:
            obj_part_sem_seg_gt = torch.as_tensor(obj_part_sem_seg_gt.astype("long"))
        
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            # The ori_size is not the real original size, but size before padding
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if obj_sem_seg_gt is not None:
                obj_sem_seg_gt = F.pad(obj_sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            
            if obj_part_sem_seg_gt is not None:
                obj_part_sem_seg_gt = F.pad(obj_part_sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        
        if obj_sem_seg_gt is not None:
            dataset_dict["sem_seg"] = obj_sem_seg_gt.long()
        
        if obj_part_sem_seg_gt is not None:
            dataset_dict["obj_part_sem_seg"] = obj_part_sem_seg_gt.long()
        
        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if obj_sem_seg_gt is not None:
            obj_sem_seg_gt = obj_sem_seg_gt.numpy()
            obj_part_sem_seg_gt = obj_part_sem_seg_gt.numpy()
            obj_instances = Instances(image_shape)
            obj_part_instances = Instances(image_shape)
            obj_classes = np.unique(obj_sem_seg_gt)
            obj_part_classes = np.unique(obj_part_sem_seg_gt)
            # remove ignored region
            obj_classes = obj_classes[obj_classes != self.ignore_label]
            
            obj_part_classes = obj_part_classes[obj_part_classes != self.ignore_label]
            obj_part_instances.gt_classes = torch.tensor(obj_part_classes, dtype=torch.int64)

            obj_masks = []
            for class_id in obj_classes:
                obj_masks.append(obj_sem_seg_gt == class_id)

            if len(obj_masks) == 0:
                # Some image does not have annotation (all ignored)
                obj_instances.gt_masks = torch.zeros((1, obj_sem_seg_gt.shape[-2], obj_sem_seg_gt.shape[-1]))
                obj_instances.gt_classes = torch.tensor([-1], dtype=torch.int64)
            else:
                obj_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in obj_masks])
                )
                obj_instances.gt_masks = obj_masks.tensor
                obj_instances.gt_classes = torch.tensor(obj_classes, dtype=torch.int64)
            
            obj_part_masks = []
            for class_id in obj_part_classes:
                obj_part_masks.append(obj_part_sem_seg_gt == class_id)

            if len(obj_part_masks) == 0:
                # Some image does not have annotation (all ignored)
                obj_part_instances.gt_masks = torch.zeros((0, obj_part_sem_seg_gt.shape[-2], obj_part_sem_seg_gt.shape[-1]))
            else:
                obj_part_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in obj_part_masks])
                )
                obj_part_instances.gt_masks = obj_part_masks.tensor
            
            dataset_dict["instances"] = obj_instances
            dataset_dict["obj_part_instances"] = obj_part_instances
        
        return dataset_dict
