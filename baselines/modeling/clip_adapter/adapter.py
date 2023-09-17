from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import BitMasks
from .utils import build_clip_model, crop_with_mask, CLIP, crop_with_bbox
from baselines.third_party import clip
from .text_prompt import build_modified_clip_model
from .text_prompt import PromptExtractor
import numpy as np
from einops import rearrange


class ClipAdapter(nn.Module):
    def __init__(self, clip_model_name: str, prompt_learner: PromptExtractor):
        super().__init__()
        self.clip_model = build_clip_model(clip_model_name)
        self.prompt_learner = prompt_learner
        self.prompt_learner.init_buffer(self.clip_model)
        self.text_feature_buffer = {}

    def forward(self, image: torch.Tensor, text: List[str], **kwargs):
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)

    def _preprocess_image(self, image: torch.Tensor):
        return image

    def _get_text_features(self, noun_list: List[str]):
        if not self.prompt_learner.with_trainable_params:

            left_noun_list = [
                noun for noun in noun_list if noun not in self.text_feature_buffer
            ]
            if len(left_noun_list) > 0:
                left_text_features = self.prompt_learner(
                    left_noun_list, self.clip_model
                )
                self.text_feature_buffer.update(
                    {
                        noun: text_feature
                        for noun, text_feature in zip(
                            left_noun_list, left_text_features
                        )
                    }
                )
            return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])
        else:
            text_features = self.prompt_learner(noun_list, self.clip_model)
            self.text_feature_buffer.update(
                {
                    noun: text_feature.detach()
                    for noun, text_feature in zip(noun_list, text_features)
                }
            )
            return text_features

    def get_text_features(self, noun_list: List[str]):
        return self._get_text_features(noun_list)

    def get_image_features(self, image: torch.Tensor):
        image_features = self.clip_model.visual(image) ### part mask 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T

    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)

class MaskFormerClipAdapter(ClipAdapter):
    def __init__(
        self,
        clip_model_name: str,
        prompt_learner: PromptExtractor,
        mask_fill: str = "mean",
        mask_expand_ratio: float = 1.0,
        mask_thr: float = 0.5,
        mask_matting: bool = False,
        region_resized: bool = True,
    ):
        super().__init__(clip_model_name, prompt_learner)
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )
        # for test
        self.mask_fill = mask_fill
        if self.mask_fill == "zero":
            self.mask_fill = (0.0, 0.0, 0.0)
        elif self.mask_fill == "mean":
            self.mask_fill = [255.0 * c for c in CLIP.PIXEL_MEAN]
        else:
            raise NotImplementedError(
                "Unknown mask_fill method: {}".format(self.mask_fill)
            )
        self.mask_expand_ratio = mask_expand_ratio
        self.mask_thr = mask_thr
        self.mask_matting = mask_matting
        self.region_resized = region_resized

        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def forward(
        self,
        image: torch.Tensor,
        text: List[str],
        mask: torch.Tensor,
        normalize: bool = True,
    ):
        image, valid_flag = self._preprocess_image(image, mask, normalize=normalize)
        if image is None:
            return None, valid_flag
        if isinstance(image, list):
            image_features = torch.cat(
                [self.get_image_features(image_i) for image_i in image], dim=0
            )
        else:
            image_features = self.get_image_features(image)
        text_feature = self.get_text_features(text)  # k,feat_dim
        return self.get_sim_logits(text_feature, image_features), valid_flag

    def _preprocess_image(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ):
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        bin_mask = bin_mask[valid]
        mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = [
            crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )[None, ...]
            for bbox, single_mask in zip(bboxes, mask)
        ]
        if len(regions) == 0:
            return None, valid
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
        return regions, valid

    def get_text_features(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)

class PerPixelClipAdapter(ClipAdapter):
    def __init__(self, *args, **kwargs):
        super(PerPixelClipAdapter, self).__init__(*args, **kwargs)
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def _preprocess_image(self, image: torch.Tensor):
        return (image.to(self.pixel_mean.device) - self.pixel_mean) / self.pixel_std

    def get_image_features(self, image: torch.Tensor, per_pixel: bool = False):
        if per_pixel:
            image_features = self.clip_model.visual(image, return_cls=False)  # b,h,w,c
        else:
            image_features = self.clip_model.visual(image)[:, None, None, :].expand(
                image.shape[0], 2, 2, -1
            )  # b,c
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(
        self, image: torch.Tensor, text: List[str], per_pixel: bool = True, **kwargs
    ):
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)

class MaskFormerObjPartClipAdapter(nn.Module):
    def __init__(self,
                clip_model_name: str,
                prompt_learner: PromptExtractor,
                mask_expand_ratio: float = 1.0,
                mask_thr: float = 0.5,
                region_resized: bool = True,) -> None:
        super().__init__()
        
        self.clip_model = build_modified_clip_model(clip_model_name)
        self.prompt_learner = prompt_learner
        # self.prompt_learner.init_buffer(self.clip_model)
        # self.prompt_learner.init_prompt(self.clip_model, "a photo of a")
        self.feature_resolution = [24, 24]
        self.clip_resolution = (384, 384)

        self.mask_expand_ratio = mask_expand_ratio
        self.mask_thr = mask_thr
        self.region_resized = region_resized
        self.fuse_weight = prompt_learner.fuse_weight
        
        self.register_buffer("pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(CLIP.PIXEL_STD).view(1, -1, 1, 1), False)
        
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )
    
    def forward(self, image: torch.Tensor, 
                text: List[str],
                obj_mask: torch.Tensor,
                part_mask: torch.Tensor,
                normalize: bool = True,
                condition: bool = False,
                return_text=False,
                return_img=False):
        image, obj_mask, part_mask, valid_flag = self._preprocess_image(image, obj_mask, part_mask, normalize=normalize)
        if image is None:
            if return_text:
                return self.get_text_features(text)
            return None, valid_flag
        if isinstance(image, list):
            image_features = torch.cat(
                [self.get_image_features(image_i) for image_i in image], dim=0
            )
        else:
            # print(image.shape, part_mask.shape)
            obj_features, part_features = self.get_image_features(image, part_mask, obj_mask)
        image_features = part_features * self.fuse_weight + obj_features * (1-self.fuse_weight)
        if return_img:
            return image_features, valid_flag
        if condition:
            text_feature = self.get_text_features(text, obj_feat=obj_features, part_feat=part_features)
        else:
            text_feature = self.get_text_features(text) # k,feat_dim
        if return_text:
                return text_feature
        
        return self.get_sim_logits(text_feature, image_features), valid_flag
    
    def _preprocess_image(
        self, image: torch.Tensor, obj_mask: torch.Tensor, part_mask: torch.Tensor, normalize: bool = True
    ):
        
        # 
        dtype = part_mask.dtype
        bin_obj_mask = obj_mask > self.mask_thr 
      
        bin_part_mask = part_mask > self.mask_thr 
        bin_part_mask[:,bin_obj_mask[0]==False] = False
            
        valid = bin_part_mask.sum(dim=(-1, -2)) > 0
        bin_part_mask = bin_part_mask[valid]
        # if bin_obj_mask.size(0) != bin_part_mask.size(0):
        #     bin_obj_mask = bin_obj_mask.repeat(bin_part_mask.size(0), 1, 1)
        # assert len(bin_obj_mask) == len(bin_part_mask)
        
        bin_obj_mask_ = BitMasks(bin_obj_mask)
        bboxes = bin_obj_mask_.get_bounding_boxes()

        ### crop object regions
        # import pdb; pdb.set_trace()
        obj_image_region = crop_with_bbox(
                image.type(dtype),
                bboxes.tensor.numpy()[0],
                expand_ratio=self.mask_expand_ratio,
            )
        obj_mask_region = crop_with_bbox(
                bin_obj_mask.type(dtype),
                bboxes.tensor.numpy()[0],
                expand_ratio=self.mask_expand_ratio,
        )
        part_mask_regions = crop_with_bbox(
                bin_part_mask.type(dtype),
                bboxes.tensor.numpy()[0],
                expand_ratio=self.mask_expand_ratio,
        )
        if len(part_mask_regions) == 0:
            return None, None, None, valid
        # from PIL import Image
        # mask_image = Image.fromarray(np.array(obj_image_region.permute(1,2,0).cpu().numpy(),dtype=np.uint8))
        # file_name = 'debug'
        # mask_image.save(f'{file_name}_crop_img.png')
        # obj_mask_region[obj_mask_region==0] = 255
        # mask_image = Image.fromarray(np.array(obj_mask_region[0].cpu().numpy(),dtype=np.uint8))
        # file_name = 'debug'
        # mask_image.save(f'{file_name}_obj.png')
        # part_mask_region[part_mask_region==0] = 255
        # for i in range(len(part_mask_region)):
        #     mask_image = Image.fromarray(np.array(part_mask_region[i].cpu().numpy(),dtype=np.uint8))
        #     file_name = f'debug_{i}'
        #     mask_image.save(f'{file_name}_part.png')
        # # exit()
        
        if normalize:
            obj_image_region = (obj_image_region / 255.0 - self.pixel_mean) / self.pixel_std
        
        if self.region_resized:
            # print(obj_image_region.shape, obj_mask_region.shape, part_mask_regions.shape)
            obj_image_region = F.interpolate(obj_image_region,
                                       size=self.clip_resolution,
                                       mode='bilinear',
                                       align_corners=False,)
            obj_mask_region = F.interpolate(obj_mask_region.unsqueeze(0),
                                       size=self.clip_resolution,
                                       mode='nearest',
                                       )
            part_mask_regions = F.interpolate(part_mask_regions.unsqueeze(1),
                                       size=self.clip_resolution,
                                       mode='nearest',
                                       )
            
        
        regions = obj_image_region.repeat(part_mask_regions.size(0), 1, 1, 1)
        obj_mask_regions = obj_mask_region.repeat(part_mask_regions.size(0), 1, 1, 1)

        return regions, obj_mask_regions, part_mask_regions, valid
    
    def get_image_features(self, image:torch.Tensor, mask: torch.Tensor, obj_mask: torch.Tensor):
        img_feat = self.clip_model.encode_image(image, dense=True)
        img_feat = rearrange(img_feat[:, 1:, :],
                             "b (h w) c->b c h w",
                             h=self.feature_resolution[0],
                             w=self.feature_resolution[1])
        img_feat = F.normalize(img_feat, dim=1)
        
        masks_resized = F.max_pool2d(mask, 16, 16)
        img_feat_masked = img_feat * masks_resized # [N, C, H, W]
        N, C, H, W = img_feat_masked.shape
        part_feat = img_feat_masked.reshape(N, C, H * W).sum(dim=2) / (masks_resized.reshape(N, 1, H * W).sum(dim=2) + 1e-6)  # [N, C]

        obj_masks_resized = F.max_pool2d(obj_mask, 16, 16)
        obj_img_feat_masked = img_feat * obj_masks_resized # [N, C, H, W]
        N, C, H, W = obj_img_feat_masked.shape
        obj_feat = obj_img_feat_masked.reshape(N, C, H * W).sum(dim=2) / (obj_masks_resized.reshape(N, 1, H * W).sum(dim=2) + 1e-6)  # [N, C]

        # return (part_feat + obj_feat) / 2
        return obj_feat, part_feat
    
    def _get_text_features(self, noun_list: List[str], obj_feat=None, part_feat=None):
        obj_name_list = [
            c.strip().split('\'s')[0] for c in noun_list
        ]
        part_name_list = [
            c.strip().split('\'s')[1][1:] for c in noun_list
        ]
        text_feat = self.prompt_learner(obj_name_list,
                                        part_name_list,
                                        self.clip_model,
                                        obj_feat,
                                        part_feat
                                        )  # [num_classes, C]
        text_feat = F.normalize(text_feat, dim=-1)

        return text_feat
    
    def get_text_features(self, noun_list: List[str], obj_feat=None, part_feat=None):
        # import pdb; pdb.set_trace()
        object_text_features = self._get_text_features(noun_list, obj_feat, part_feat)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=-2)
        
    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T
    
    def get_batch_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * torch.einsum('blc,bnc->bln', image_features, text_features)
    
    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)

        


