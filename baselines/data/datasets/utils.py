import json
import logging
import os

from detectron2.data.datasets.coco import load_sem_seg

logger = logging.getLogger(__name__)

def load_obj_part_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg", data_list=None):
    data_dicts = load_sem_seg(gt_root, image_root, gt_ext, image_ext)
    if data_list is not None:
        img_list = json.load(open(data_list,'r'))
        img_list = [item["file_name"] for item in img_list]
    new_data_dicts = []
    for i,data in enumerate(data_dicts):
        if data_list is not None:
            if data["file_name"] not in img_list:
                continue
        data_dicts[i]["obj_sem_seg_file_name"] = data["sem_seg_file_name"].replace('part','obj')
        new_data_dicts.append(data_dicts[i])
    return new_data_dicts


def load_binary_mask(gt_root, image_root, gt_ext="png", image_ext="jpg", label_count="_part_label_count.json", base_classes=None):
    """
    Flatten the results of `load_sem_seg` to annotations for binary mask.

    `label_count_file` contains a dictionary like:
    ```
    {
        "xxx.png":[0,3,5],
        "xxxx.png":[3,4,7],
    }
    ```
    """
    label_count_file = gt_root + label_count
    with open(label_count_file) as f:
        label_count_dict = json.load(f)

    data_dicts = load_sem_seg(gt_root, image_root, gt_ext, image_ext)
    flattened_data_dicts = []
    for data in data_dicts:
        data['obj_sem_seg_file_name'] = data["sem_seg_file_name"].replace('_part','_obj')
        category_per_image = label_count_dict[
            os.path.basename(data["sem_seg_file_name"])
        ]
        if base_classes is not None:
            category_per_image = [i for i in category_per_image if i in base_classes]
        flattened_data = [
            dict(**{"category_id": cat}, **data) for cat in category_per_image
        ]
        flattened_data_dicts.extend(flattened_data)
    logger.info(
        "Loaded {} images with flattened semantic segmentation from {}".format(
            len(flattened_data_dicts), image_root
        )
    )
    return flattened_data_dicts
