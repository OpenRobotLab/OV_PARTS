from typing import List, Tuple

# import clip
from baselines.third_party import clip
import torch
from torch import nn

from .utils import CLIP
from baselines.modeling.transformer.transformer_predictor import MLP
from detectron2.utils.comm import get_local_rank, synchronize
from einops import rearrange


class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False
        self.with_trainable_params = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError()


class PredefinedPromptExtractor(PromptExtractor):
    def __init__(self, templates: List[str]):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


class ImageNetPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(CLIP.IMAGENET_PROMPT)


class VILDPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(CLIP.VILD_PROMPT)


class LearnablePromptExtractor(PromptExtractor):
    def __init__(self, prompt_dim: int, prompt_shape: Tuple[int, int]):
        super().__init__()
        assert len(prompt_shape) == 2, "prompt_shape must be a tuple of length 2"
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self._buffer_init = False
        self.with_trainable_params = True

    def _init_prompt(self, length):
        if length == 0:
            return None
        prompt_tensor = torch.empty(length, self.prompt_dim)
        nn.init.normal_(prompt_tensor, std=0.02)
        return nn.Parameter(prompt_tensor)

    def init_buffer(self, clip_model):
        sentence = "X."
        prompt = clip.tokenize(sentence)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(
                clip_model.dtype
            )  # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512
        self.noun_bucket = {}
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        if not self._buffer_init:
            raise RuntimeError(
                f"Buffer of {self.__class__.__name__} is not initialized"
            )
        self._update_noun_features(noun_list, clip_model)

        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)
        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix)
        # only process those which are not in bucket
        lengths = [
            len(prefix) + len(suffix) + len(self.noun_bucket[noun])
            for noun in noun_list
        ]
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.noun_bucket[noun], suffix]
                    + [self.pad_signal.expand(77 - length, -1)]
                )
                for noun, length in zip(noun_list, lengths)
            ]
        )  # cls,77,512
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1
        text_features = self.get_text_feature(embeddings, indices, clip_model)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _update_noun_features(self, noun_list, clip_model):
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        if len(left_class_names) > 0:
            with torch.no_grad():
                tokens, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [
                    n - 2 for n in name_lengths
                ]  # remove start end end prompt
                text_embeddings = clip_model.token_embedding(
                    tokens.to(self.device)
                ).type(clip_model.dtype)
                text_embeddings = [
                    embedding[1 : 1 + length]
                    for embedding, length in zip(text_embeddings, name_lengths)
                ]
            self.noun_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, text_embeddings)
                }
            )

    @staticmethod
    def get_text_feature(x, indices, clip_model):
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix},dimension:{self.prompt_dim}\n"
        repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr

class LearnablePartPromptExtractor(nn.Module):
    def __init__(self,
                 prompt_dim,
                 prompt_shape):
        super().__init__()
        assert len(prompt_shape) == 3
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self._buffer_init = False
        self.with_trainable_params = True   
        self.fuse_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def _init_prompt(self, length, init_text=None, clip_model=None):
        if length == 0:
            return None
        if init_text is None:
            prompt_tensor = torch.empty(length, self.prompt_dim)
            nn.init.normal_(prompt_tensor, std=0.02)
        else:
            assert clip_model is not None
            init_text = init_text.replace("_", " ")
            assert len(init_text.split(" ")) == length
            prompt = clip.tokenize(init_text)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(clip_model.dtype)
            prompt_tensor = embedding[0, 1 : 1 + length, :]
        return nn.Parameter(prompt_tensor)
    
    # def init_metanet(self):
    #     self.metanet = MLP(self.prompt_dim, self.prompt_dim, self.prompt_dim, 2)

    
    # def init_fuse_weight(self, fuse_weight=torch.tensor(0.0)):
    #     self.fuse_weight = nn.Parameter(fuse_weight, requires_grad=True)
    
    def init_prompt(self, clip_model, text=None):
        # TODO init prompt
        self.prefix_prompt = self._init_prompt(self.n_prefix, init_text=text, clip_model=clip_model)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self.connect_prompt = self._init_prompt(self.n_connect)

    def init_buffer(self, clip_model):
        sentence = "'s."
        prompt = clip.tokenize(sentence)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(
                clip_model.dtype
            )  # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("ss_signal", embedding[0, 1:2, :])  # 1, 512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512
        self.noun_bucket = {}
        self._buffer_init = True
    
    

    def forward(self,
                obj_name_list: List[str],
                part_name_list: List[str],
                clip_model: nn.Module,
                obj_feat=None,
                part_feat=None):
        if not self._buffer_init:
            raise RuntimeError(f"Buffer of {self.__class__.__name__} is not initialized")
        
        # TODO update noun embeddings here
        self._update_noun_features(obj_name_list, clip_model)
        self._update_noun_features(part_name_list, clip_model)
        
        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)
        
        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix)
        
        middle = [self.connect_prompt]
        middle = torch.cat(middle)
        
        lengths = [len(prefix) + len(suffix) + len(self.noun_bucket[obj_name]) + len(self.noun_bucket[part_name]) + len(middle)
                   for obj_name, part_name in zip(obj_name_list, part_name_list)]
        if obj_feat is not None:
            # import pdb; pdb.set_trace()
            # obj_feat = self.metanet(obj_feat)
            # prefix = prefix.unsqueeze(0).repeat(obj_feat.size(0), 1, 1)
            prefix[1:,:] += obj_feat#.unsqueeze(1)
            middle += obj_feat

            
        # if part_feat is not None:
        #     part_feat = self.metanet(part_feat)
        #     # middle = middle.unsqueeze(0).repeat(obj_feat.size(0), 1, 1)
        #     middle += part_feat#.unsqueeze(1)
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.noun_bucket[obj_name], middle, self.noun_bucket[part_name], suffix]
                    + [self.pad_signal.expand(77 - length, -1)]
                )
                for obj_name, part_name, length in zip(obj_name_list, part_name_list, lengths)
            ]
        )  # cls,77,512

        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1

        text_features = self.get_text_feature(embeddings.half(), indices, clip_model)
        
        return text_features.float()

    def _update_noun_features(self, noun_list, clip_model):
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        if len(left_class_names) < 1:
            return
        with torch.no_grad():
            tokens, name_lengths = clip.tokenize(left_class_names, return_length=True)
            name_lengths = [n - 2 for n in name_lengths]
            text_embeddings = clip_model.token_embedding(tokens.to(self.device)).type(clip_model.dtype)
            text_embeddings = [embedding[1: 1+length] for embedding, length in zip(text_embeddings, name_lengths)]
            self.noun_bucket.update({
                name: embedding for name, embedding in zip(left_class_names, text_embeddings)
            })

    @staticmethod
    def get_text_feature(x, indices, clip_model):
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[-1]

    @property
    def n_connect(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device


def build_modified_clip_model(model: str,
                              frozen: bool = True):
    rank = get_local_rank()
    if rank == 0:
        model, _ = clip.load(model, device='cpu', jit=False, prompt_depth=0, prompt_length=0)
    synchronize()
    if rank != 0:
        model, _ = clip.load(model, device='cpu', jit=False, prompt_depth=0, prompt_length=0)
    synchronize()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    return model
