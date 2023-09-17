import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .coco import load_coco_json
from .utils import load_binary_mask
import json

OBJ_CLASS_NAMES = ['airplane', 'armchair', 'bed', 'bench', 'bookcase', 'bus', 'cabinet', 'car', 'chair', 'chandelier', 'chest of drawers', 'clock', 'coffee table', 'computer', 'cooking stove', 'desk', 'dishwasher', 'door', 
                   'fan', 'glass', 'kitchen island', 'lamp', 'light', 'microwave', 'minibike', 'ottoman', 'oven', 'person', 'pool table', 'refrigerator', 'sconce', 'shelf', 'sink', 'sofa', 'stool', 
                   'swivel chair', 'table', 'television receiver', 'toilet', 'traffic light', 'truck', 'van', 'wardrobe', 'washer']
CLASS_NAMES = ["person's arm", "person's back", "person's foot", "person's gaze", "person's hand", "person's head", "person's leg", "person's neck", "person's torso", "door's door frame", "door's handle", "door's knob", 
               "door's panel", "clock's face", "clock's frame", "toilet's bowl", "toilet's cistern", "toilet's lid", "cabinet's door", "cabinet's drawer", "cabinet's front", "cabinet's shelf", 
               "cabinet's side", "cabinet's skirt", "cabinet's top", "sink's bowl", "sink's faucet", "sink's pedestal", "sink's tap", "sink's top", "lamp's arm", "lamp's base", "lamp's canopy", "lamp's column", 
               "lamp's cord", "lamp's highlight", "lamp's light source", "lamp's shade", "lamp's tube", "sconce's arm", "sconce's backplate", "sconce's highlight", "sconce's light source", "sconce's shade", "chair's apron",
               "chair's arm", "chair's back", "chair's base", "chair's leg", "chair's seat", "chair's seat cushion", "chair's skirt", "chair's stretcher", "chest of drawers's apron", "chest of drawers's door", "chest of drawers's drawer", 
               "chest of drawers's front", "chest of drawers's leg", "chandelier's arm", "chandelier's bulb", "chandelier's canopy", "chandelier's chain", "chandelier's cord", "chandelier's highlight", "chandelier's light source", "chandelier's shade",
               "bed's footboard", "bed's headboard", "bed's leg", "bed's side rail", "table's apron", "table's drawer", "table's leg", "table's shelf", "table's top", "table's wheel", "armchair's apron", "armchair's arm", "armchair's back", 
               "armchair's back pillow", "armchair's leg", "armchair's seat", "armchair's seat base", "armchair's seat cushion", "ottoman's back", "ottoman's leg", "ottoman's seat", "shelf's door", "shelf's drawer", "shelf's front", "shelf's shelf", 
               "swivel chair's back", "swivel chair's base", "swivel chair's seat", "swivel chair's wheel", "fan's blade", "fan's canopy", "fan's tube", "coffee table's leg", "coffee table's top", "stool's leg", "stool's seat", "sofa's arm", "sofa's back", 
               "sofa's back pillow", "sofa's leg", "sofa's seat base", "sofa's seat cushion", "sofa's skirt", "computer's computer case", "computer's keyboard", "computer's monitor", "computer's mouse", "desk's apron", "desk's door", "desk's drawer", "desk's leg",
               "desk's shelf", "desk's top", "wardrobe's door", "wardrobe's drawer", "wardrobe's front", "wardrobe's leg", "wardrobe's mirror", "wardrobe's top", "car's bumper", "car's door", "car's headlight", "car's hood", "car's license plate", "car's logo", 
               "car's mirror", "car's wheel", "car's window", "car's wiper", "bus's bumper", "bus's door", "bus's headlight", "bus's license plate", "bus's logo", "bus's mirror", "bus's wheel", "bus's window", "bus's wiper", "oven's button panel", "oven's door", 
               "oven's drawer", "oven's top", "cooking stove's burner", "cooking stove's button panel", "cooking stove's door", "cooking stove's drawer", "cooking stove's oven", "cooking stove's stove", "microwave's button panel", "microwave's door", "microwave's front",
               "microwave's side", "microwave's top", "microwave's window", "refrigerator's button panel", "refrigerator's door", "refrigerator's drawer", "refrigerator's side", "kitchen island's door", "kitchen island's drawer", "kitchen island's front", "kitchen island's side", 
               "kitchen island's top", "dishwasher's button panel", "dishwasher's handle", "dishwasher's skirt", "bookcase's door", "bookcase's drawer", "bookcase's front", "bookcase's side", "television receiver's base", "television receiver's buttons", "television receiver's frame",
               "television receiver's keys", "television receiver's screen", "television receiver's speaker", "glass's base", "glass's bowl", "glass's opening", "glass's stem", "pool table's bed", "pool table's leg", "pool table's pocket", "van's bumper", "van's door", "van's headlight", 
               "van's license plate", "van's logo", "van's mirror", "van's taillight", "van's wheel", "van's window", "van's wiper", "airplane's door", "airplane's fuselage", "airplane's landing gear", "airplane's propeller", "airplane's stabilizer", "airplane's turbine engine", 
               "airplane's wing", "truck's bumper", "truck's door", "truck's headlight", "truck's license plate", "truck's logo", "truck's mirror", "truck's wheel", "truck's window", "minibike's license plate", "minibike's mirror", "minibike's seat", "minibike's wheel", "washer's button panel", 
               "washer's door", "washer's front", "washer's side", "bench's arm", "bench's back", "bench's leg", "bench's seat", "traffic light's housing", "traffic light's pole", "light's aperture", "light's canopy", "light's diffusor", "light's highlight", "light's light source", "light's shade"]
OBJ_NOVEL_CLASS_NAMES = ['bench', 'bus', 'fan', 'desk', 'stool', 'truck', 'van', 'swivel chair', 'oven', 'ottoman', 'kitchen island']
OBJ_BASE_CLASS_NAMES = [c for c in OBJ_CLASS_NAMES if c not in OBJ_NOVEL_CLASS_NAMES]
BASE_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if c.split('\'s')[0] not in OBJ_NOVEL_CLASS_NAMES]
obj_map = {OBJ_CLASS_NAMES.index(c): i for i,c in enumerate(OBJ_BASE_CLASS_NAMES)}
obj_part_map = {CLASS_NAMES.index(c): i for i,c in enumerate(BASE_CLASS_NAMES)}

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "ade_obj_part_sem_seg_train": (
        "images/training",
        "ade20k_instance_train.json",
    ),
    "ade_obj_part_sem_seg_val": (
        "images/validation",
        "ade20k_instance_val.json",
    )
}

def _get_obj_part_meta(cat_list, obj_list):
    ret = {
        "stuff_classes": cat_list,
        "obj_classes": obj_list
    }
    return ret

def load_json(_root, image_root, json_file, extra_annotation_keys=None, per_image=False, val_all=False, data_list=None):
    dataset_dicts = []
    obj_dataset_dicts = load_coco_json(os.path.join(_root, json_file), os.path.join(_root, image_root), extra_annotation_keys=extra_annotation_keys)
    if data_list is not None:
        img_list = json.load(open(data_list,'r'))
        img_list = [item["file_name"] for item in img_list]
    for dataset_dict in obj_dataset_dicts:
        if data_list is not None:
            if dataset_dict["file_name"] not in img_list:
                continue
        if not per_image:
            for anno in dataset_dict['annotations']:
                if 'part_category_id' in anno:
                    part_ids = anno['part_category_id']
                    for part_id in part_ids:
                        record = {}
                        record['file_name'] = dataset_dict['file_name']
                        record['height'] = dataset_dict['height']
                        record['width'] = dataset_dict['width']
                        record['obj_annotations'] = anno
                        record["obj_sem_seg_file_name"] = 'NA'
                        record['category_id'] = part_id
                        record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations_detectron2_part').replace('jpg','png')
                        dataset_dicts.append(record)
                else:
                    record = {}
                    record['file_name'] = dataset_dict['file_name']
                    record['height'] = dataset_dict['height']
                    record['width'] = dataset_dict['width']
                    record['obj_annotations'] = [anno]
                    record["obj_sem_seg_file_name"] = 'NA'
                    record['category_id'] = anno['category_id']
                    record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations_detectron2_part').replace('jpg','png')
                    if val_all and anno['category_id'] in obj_map:
                        continue
                    dataset_dicts.append(record)
        else:
            record = {}
            record['file_name'] = dataset_dict['file_name']
            record['height'] = dataset_dict['height']
            record['width'] = dataset_dict['width']
            record['obj_annotations'] = dataset_dict['annotations']
            record["obj_sem_seg_file_name"] = 'NA'
            # record['category_id'] = anno['category_id']
            record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations_detectron2_part').replace('jpg','png')
            dataset_dicts.append(record)
    return dataset_dicts

def load_train_val_json(_root, image_root, val_json_file, few_shot=False, data_list=None):
    val_dataset_dicts = load_json(_root, image_root, val_json_file)
    train_dataset_dicts = load_json(_root, image_root.replace('validation','training'), val_json_file.replace('val','train'), val_all=True)
    
    if few_shot:
        assert data_list is not None
        train_fw = json.load(open(data_list,'r'))
        train_fw_file_names = [t['file_name'] for t in train_fw]
        tmp = []
        for record in train_dataset_dicts:
            if record['file_name'] not in train_fw_file_names:
                tmp.append(record)
        train_dataset_dicts = tmp

    return val_dataset_dicts + train_dataset_dicts
        
def register_ade20k_part_234(root):
    root = os.path.join(root, "ADE20KPart234")
    meta = _get_obj_part_meta(CLASS_NAMES, OBJ_CLASS_NAMES)
    base_meta = _get_obj_part_meta(BASE_CLASS_NAMES, OBJ_BASE_CLASS_NAMES)
    for name, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        all_name = name
        name_obj_cond = all_name + "_obj_condition"
        name_few_shot = all_name + "_few_shot"
        if 'train' in name:
            DatasetCatalog.register(all_name, lambda: load_json(root, "images/training", "ade20k_instance_train.json",per_image=True)) ### image level annotations
            DatasetCatalog.register(name_obj_cond, lambda: load_json(root, "images/training", "ade20k_instance_train.json")) ### object instance level annotations
            DatasetCatalog.register(name_few_shot, lambda: load_json(root, "images/training", "ade20k_instance_train.json", data_list=f'{root}/train_16shot.json')) ### object instance level annotations in few shot
            MetadataCatalog.get(all_name).set(
                image_root=image_root,
                sem_seg_root=image_root,
                evaluator_type="sem_seg",
                ignore_label=65535,
            )
        else:
            DatasetCatalog.register(name_obj_cond, lambda: load_train_val_json(root, "images/validation", "ade20k_instance_val.json")) ### test on both excusive train and val set
            DatasetCatalog.register(name_few_shot, lambda: load_train_val_json(root, "images/validation", "ade20k_instance_val.json", few_shot=True, data_list=f'{root}/train_16shot.json')) ### few shot test set
        
        MetadataCatalog.get(name_obj_cond).set(
            image_root=image_root,
            sem_seg_root=image_root,
            evaluator_type="sem_seg",
            ignore_label=65535,
        )
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
        
        if 'train' in name:
            MetadataCatalog.get(name_obj_cond).set(**base_meta)
            MetadataCatalog.get(name_obj_cond).set(obj_map=obj_map)
            MetadataCatalog.get(name_obj_cond).set(obj_part_map=obj_part_map)
            MetadataCatalog.get(all_name).set(**base_meta)
            MetadataCatalog.get(all_name).set(obj_map=obj_map)
            MetadataCatalog.get(all_name).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(name_obj_cond).set(**meta)
            
        MetadataCatalog.get(name_few_shot).set(**meta)
        MetadataCatalog.get(name_few_shot).set(
            image_root=image_root,
            sem_seg_root=image_root,
            evaluator_type="sem_seg",
            ignore_label=65535,
        )
        
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
        MetadataCatalog.get(name_few_shot).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        
    
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_part_234(_root)
