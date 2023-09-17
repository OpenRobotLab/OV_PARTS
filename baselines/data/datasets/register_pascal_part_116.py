# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .utils import load_binary_mask, load_obj_part_sem_seg

OBJ_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
OBJ_BASE_CLASS_NAMES = [
    c for i, c in enumerate(OBJ_CLASS_NAMES) if c not in ["bird", "car", "dog", "sheep", "motorbike"]
]
CLASS_NAMES = ["aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", 
               "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", "bottle's body", "bottle's cap", "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", 
               "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", 
               "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn", 
               "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", 
               "horse's muzzle", "horse's hoof", "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", 
               "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand","pottedplant's pot", "pottedplant's plant", 
               "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", 
               "train's coach", "tvmonitor's screen"]

BASE_CLASS_NAMES = [
    c for i, c in enumerate(CLASS_NAMES) if c not in ["bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot",
                                                      "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window",
                                                      "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle",
                                                      "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn",
                                                      "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight"]
]
obj_map = {CLASS_NAMES.index(c): i for i,c in enumerate(BASE_CLASS_NAMES)}
obj_part_map = {CLASS_NAMES.index(c): i for i,c in enumerate(BASE_CLASS_NAMES)}

def _get_voc_obj_part_meta(cat_list, obj_list):
    ret = {
        "stuff_classes": cat_list,
        "obj_classes": obj_list,
        "obj_base_classes": OBJ_BASE_CLASS_NAMES
    }
    return ret


def register_pascal_part_116(root):
    root = os.path.join(root, "PascalPart116")
    meta = _get_voc_obj_part_meta(CLASS_NAMES, OBJ_CLASS_NAMES)
    base_meta = _get_voc_obj_part_meta(BASE_CLASS_NAMES, OBJ_BASE_CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations_detectron2_part/train"),
        ("val", "images/val", "annotations_detectron2_part/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        
        ################################ part sem seg without object sem seg ############################
        all_name = f"voc_obj_part_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_obj_part_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
            )
        if name == 'train':
            MetadataCatalog.get(all_name).set(**base_meta)
            MetadataCatalog.get(all_name).set(obj_map=obj_map)
            MetadataCatalog.get(all_name).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(all_name).set(**meta)
        MetadataCatalog.get(all_name).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        ################################ part sem seg with object sem seg ############################
        name_obj_cond = all_name + "_obj_condition"
        DatasetCatalog.register(
            name_obj_cond,
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg", label_count="_obj_label_count.json"
            ),
        )
        MetadataCatalog.get(name_obj_cond).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
            )
        if name == 'train':
            MetadataCatalog.get(name_obj_cond).set(**base_meta)
            MetadataCatalog.get(name_obj_cond).set(obj_map=obj_map)
            MetadataCatalog.get(name_obj_cond).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(name_obj_cond).set(**meta)
        MetadataCatalog.get(name_obj_cond).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        ################################ part sem seg in few shot setting ############################
        if name == 'train':
            name_few_shot = all_name + "_few_shot"
            DatasetCatalog.register(
                name_few_shot,
                lambda x=image_dir, y=gt_dir: load_obj_part_sem_seg(
                    y, x, gt_ext="png", image_ext="jpg", data_list=f'{root}/train_16shot.json'
                ),
            )
            MetadataCatalog.get(name_few_shot).set(
                    image_root=image_dir,
                    sem_seg_root=gt_dir,
                    evaluator_type="sem_seg",
                    ignore_label=255,
                )

            MetadataCatalog.get(name_few_shot).set(**meta)
            MetadataCatalog.get(name_few_shot).set(
                evaluation_set={
                    "base": [
                        meta["stuff_classes"].index(n) for n in meta["stuff_classes"]
                    ],
                },
                trainable_flag=[1] * len(meta["stuff_classes"]),
            )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_part_116(_root)

