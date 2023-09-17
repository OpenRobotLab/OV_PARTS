## OV-PARTS: Towards Open-Vocabulary Part Segmentation

This codebase contains code for baselines used in the paper "OV-PARTS: Towards Open-Vocabulary Part Segmentation".
![](assets/ov_parts.jpg)

- ### Enviroment
    ```bash
    torch==1.13.1
    torchvision==0.14.1
    detectron2==0.6 #Following https://detectron2.readthedocs.io/en/latest/tutorials/install.html to install it and some required packages
    mmcv==1.7.1
    ```
    FurtherMore, install the modified clip package.
    ```bash
    cd third_party/CLIP
    python -m pip install -e .
    ```

- ### Data Preparation
    We provide the download links for the two benchmark datasets in OV-PARTS: the refined Pascal-Part-116 and ADE20K-Part-234 datasets.

    [[Pascal-Part-116]](https://drive.google.com/file/d/1f5kqrM2_iK_bWmQBW3rdSnGrnke4PUbX/view)
    [[ADE20K-Part-234]](https://drive.google.com/file/d/1EBVPW_tqzBOQ_DC6yLcouyxR7WrctRKi/view)

    After downloading the datasets, please extract the files by running the following command and place the extracted folder under the "Datasets" directory.
    ```bash
    tar -xzf PascalPart116.tar.gz
    tar -xzf ADE20KPart234.tar.gz
    ```
    The Datasets folder should follow this structure:
    ```shell
    Datasets/
    ├─Pascal-Part-116/
    │ ├─train_16shot.json
    │ ├─images/
    │ │ ├─train/
    │ │ └─val/
    │ ├─annotations_detectron2_obj/
    │ │ ├─train/
    │ │ └─val/
    │ └─annotations_detectron2_part/
    │   ├─train/
    │   └─val/
    └─ADE20K-Part-234/
      ├─images/
      │ ├─training/
      │ ├─validation/
      ├─train_16shot.json
      ├─ade20k_instance_train.json
      ├─ade20k_instance_val.json
      └─annotations_detectron2_part/
        ├─training/
        └─validation/
    ```

- ### Training
    - Training the two-stage baseline ZSseg+.

      Please first download the clip model fintuned with [CPTCoOp](https://drive.google.com/drive/folders/1ME6cVMiYE5Z2yZscP_jj2g1NwamIApw1?usp=sharing).

      Then run the training command:
      ```bash
      # For ZSSeg+.
      python train_net.py --num-gpus 8 --config-file configs/${SETTING}/zsseg+_R50_coop_${DATASET}.yaml
      ```

    - Training the one-stage baselines CLIPSeg and CATSeg. 

      Please first download the pre-trained object models of CLIPSeg and CATSeg and place them under the "pretrain_weights" directory.

      | Models | Pre-trained checkpoint |
      |:----------:|:-------------:|
      | CLIPSeg | [download](https://huggingface.co/CIDAS/clipseg-rd64-refined) |
      | CATSeg | [download](https://huggingface.co/hamacojr/CAT-Seg/blob/main/model_final_base.pth) |
      
      Then run the training command:
      ```bash
      # For CATseg.
      python train_net.py --num-gpus 8 --config-file configs/${SETTING}/catseg_${DATASET}.yaml

      # For CLIPseg.
      python train_net.py --num-gpus 8 --config-file configs/${SETTING}/clipseg_${DATASET}.yaml
      ```

    

- ### Evaluation
    We provide the trained weights for the three baseline models reported in the paper.

    | Models | Setting | Pascal-Part-116 checkpoint | ADE20K-Part-234 checkpoint |
    |:------:|:-------:|:----------:|:----------:|
    | ZSSeg+ | Zero-shot | [download](https://drive.google.com/file/d/10evrfHIRARbil5WSaziFth1aAFeky185/view) |[download](https://drive.google.com/file/d/1ayb5n-bVI0oBQxYzojj0o0yN6bnCuum8/view) |
    | CLIPSeg | Zero-shot | [download](https://drive.google.com/file/d/1WkCu3-KA2Oho5xzBXDR_HUmBvvKKYSQM/view) | [download](https://drive.google.com/file/d/1Ydh1wn1H8TPVrO24TaAIUMMLhqAPXH-W/view) |
    | CatSet | Zero-shot | [download](https://drive.google.com/file/d/11OOO5NmJkkeJl5oTbxbTZUgFyFXSKVRZ/view) |[download](https://drive.google.com/file/d/1jvGQtftwRuzbFnphQXftosdL_YDwMWAx/view) |
    | CLIPSeg | Few-shot | [download](https://drive.google.com/file/d/1WPDoc4igtDQ9H46Wwp4aaHXgu4gMNICO/view) |[download](https://drive.google.com/file/d/1LCCBCgRqqerf2ZYLfClm4a_hP8dWZGzv/view) |
    | CLIPSeg | cross-dataset | - |[download](https://drive.google.com/file/d/1G5SMWhZ0UwAiW2CvEDXpxPhfksN33hKp/view?usp=sharing) |

    To evaluate the trained models, add ```--eval-only``` to the training command.
    For example:
    ```bash
      python train_net.py --num-gpus 8 --config-file configs/${SETTING}/catseg_${DATASET}.yaml --eval-only MODEL.WEIGHTS ${WEIGHT_PATH}
    ```

- ### Acknowledgement
  We would like to express our gratitude to the open-source projects and their contributors, including [ZSSeg](https://github.com/MendelXu/zsseg.baseline), [CATSeg](https://github.com/KU-CVLAB/CAT-Seg) and [CLIPSeg](https://github.com/timojl/clipseg). Their valuable work has greatly contributed to the development of our codebase.

