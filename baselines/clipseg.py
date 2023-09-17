from detectron2.modeling import META_ARCH_REGISTRY
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from detectron2.data import MetadataCatalog
from detectron2.config import configurable

@META_ARCH_REGISTRY.register()
class CLIPSeg(nn.Module):
    @configurable
    def __init__(self, train_dataset, test_dataset):
        super().__init__()
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        train_class_texts = MetadataCatalog.get(train_dataset).stuff_classes
        self.train_class_texts = [c.replace('\'s', '') for c in train_class_texts]
        self.train_obj_classes = MetadataCatalog.get(train_dataset).obj_classes

        self.test_class_texts = MetadataCatalog.get(test_dataset).stuff_classes
        self.test_obj_classes = MetadataCatalog.get(test_dataset).obj_classes
        
        self.segmentation_background_threshold = 0.0
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.device = "cuda" 
        self.ignore_label = MetadataCatalog.get(test_dataset).ignore_label
        
        for name, params in self.clipseg_model.named_parameters():
            if 'clip.text_model.embeddings' in name or 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
                params.requires_grad = True
            else:
                params.requires_grad = False
        self.train_text_encoding = self.clipseg_processor.tokenizer(self.train_class_texts, return_tensors="pt", padding="max_length")
        
    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0]
        ret['test_dataset'] = cfg.DATASETS.TEST[0]
        return ret
    
    def preds_to_semantic_inds(self, preds, threshold):
        flat_preds = preds.reshape((preds.shape[0], -1))
        # Initialize a dummy "unlabeled" mask with the threshold
        flat_preds_with_treshold = torch.full(
            (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
        )
        flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

        # Get the top mask index for each pixel
        semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
            (preds.shape[-2], preds.shape[-1])
        )
        return semantic_inds
    
    def clipseg_segmentation(
        self, model, images, test_text, device
    ):
        logits = []
        input = self.clipseg_processor(images=images, return_tensors="pt").to(device)
        if self.training:
            text = self.train_text_encoding
        else:
            text=  test_text
        input.update(text)
        input = input.to(device)
        outputs = model(**input)
        logits = outputs.logits
        return logits
    
    def inference(self, batched_inputs):
        image = Image.open(batched_inputs[0]["file_name"])
        image = image.convert("RGB")
        with torch.no_grad():
            logits = self.clipseg_segmentation(
                self.clipseg_model,
                [image],
                self.clipseg_processor.tokenizer([part.replace('\'s', '') for part in self.test_class_texts], return_tensors="pt", padding="max_length"),
                self.device,
            )
            upscaled_logits = nn.functional.interpolate(
                            logits[:,:-1,:,:],
                            size=(image.size[1], image.size[0]),
                            mode="bilinear",
                            )
            clipseg_preds = torch.sigmoid(upscaled_logits)
        gt_objs = [self.test_obj_classes[i] for i in torch.unique(batched_inputs[0]["sem_seg"]) if i != self.ignore_label]
        part_category_names = []
        part_inds = []
        for obj in gt_objs:
            for i,part in enumerate(self.test_class_texts):
                if part.split('\'s')[0] == obj:
                    part_category_names.append(part.replace('\'s', ''))
                    part_inds.append(i)
        no_part_ids = [i for i in range(len(self.test_class_texts)) if i not in part_inds]  
        preds = clipseg_preds.squeeze(0)
        preds[no_part_ids] = 0.0
        results = [{"sem_seg": preds}]
        return results
    
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        # images = [Image.open(x["file_name"]).convert("RGB") for x in batched_inputs]
        images = [x["image"].to(self.device) for x in batched_inputs]
        gts = [x["obj_part_sem_seg"].to(self.device) for x in batched_inputs]
        outputs = self.clipseg_segmentation(self.clipseg_model, images, None, self.device) #[b,n,h,w]
        targets = torch.stack([nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(),
                size=(outputs.shape[-2], outputs.shape[-1]),
                mode="nearest") for gt in gts]).long().squeeze(1).squeeze(1) #[b,h,w]

        num_classes = outputs.shape[1]
        mask = targets != self.ignore_label #[b,h,w]
        outputs = outputs.permute(0,2,3,1) #[b,h,w,n]
        _targets = torch.zeros(outputs.shape, device=self.device)
        class_weight = torch.ones(num_classes).cuda()
        _targets[:,:,:,-1] = 1
        class_weight[-1] = 0.05
        _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
        _targets[mask] = _onehot

        loss = F.binary_cross_entropy_with_logits(outputs, _targets, weight=class_weight)
        losses = {"loss_sem_seg" : loss}
        return losses
        