"""
Transformer models used by the interfase of HuggingFace.

TODO: Review how it works with common loss functions now that the trainer
class squeezing of predictions and labels has been removed.
"""

import torch
import torch.nn as nn

from transformers import CLIPModel, CLIPTextModel, CLIPProcessor
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, Any

import numpy

Tensor = torch.Tensor
Tokenizer = PreTrainedTokenizer


class CustomTextualCLIP(nn.Module):
    def __init__(self, num_classes: int, finetune: bool = False, multisentence: bool = False):
        """
        Visual half of a CLIP model ready to fine-tune on classification
        and/or regression tasks.

        Args:
            num_classes: Labels to predict.
            finetune: If True, adjust only the last layer, otherwise train
            the whole model.

        """
        super(CustomTextualCLIP, self).__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        t = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.multiple = multisentence
        self.base_text_clip = t
        self.base_text_proj = clip.text_projection
        self.output_embed = True if num_classes < 1 else False
        if finetune:
            for layer in [self.base_text_clip, self.base_text_proj]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.classifier = nn.Linear(self.base_text_proj.out_features, num_classes)

    def forward(self, x: Dict) -> Tensor:
        if self.multiple:
            z = self._forward_multiple(x)
        else:
            z = self._simple(x)
        if self.output_embed:
            return z
        z = self.classifier(z)
        # Do not apply softmax if there is a single class!!
        return nn.functional.log_softmax(z, dim=1) if self.classifier.out_features > 1 else torch.sigmoid(z)

    def _forward_multiple(self, x: Dict) -> Tensor:
        """Forward pass splitting in different sentences within each sample."""
        # The input is a dict. Each value is a tensor of shape (BS, S, L) where
        # BS is the batch size, S is the number of sentences and L is the
        # length of the sentence. We need to split the sentences and pass
        # them through the model.
        xx = [
            {"input_ids": x["input_ids"][i, :, :], "attention_mask": x["attention_mask"][i, :, :]}
            for i in range(x["input_ids"].shape[0])
        ]
        # XX is a list of dictionaries. Each dictionary has two keys: input_ids
        # and attention_mask. Each value is a tensor of shape (S, L).
        z = [torch.mean(self._simple(x_i), dim=0) for x_i in xx]
        return torch.stack(z)

    def _simple(self, x: Dict) -> Tensor:
        # x = {k: x[k].squeeze() for k in x}
        # print("🚀 ~ file: transformers.py:65 ~ x:", x)
        z = self.base_text_clip(**x).pooler_output
        return self.base_text_proj(z)

    def resize_embeddings_layer(self, num_tokens: int) -> None:
        """Either augment or decrease the size of the embeddings layer of the
        underlying model.

        Args:
            num_tokens: Number of tokens in the tokenizer.

        """
        self.base_text_clip.resize_token_embeddings(num_tokens)


class CustomVisualCLIP(nn.Module):
    def __init__(self, num_classes: int, finetune: bool = False):
        """
        Visual half of a CLIP model ready to fine-tune on classification
        and/or regression tasks.

        Args:
            num_classes: Labels to predict.
            finetune: If True, adjust only the last layer, otherwise train
            the whole model.

        """
        super(CustomVisualCLIP, self).__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # from transformers import CLIPConfig
        # clip = CLIPModel(CLIPConfig())
        self.base_visual_clip = clip.vision_model
        self.base_visual_proj = clip.visual_projection
        self.output_embed = True if num_classes < 1 else False
        if finetune:
            for layer in [self.base_visual_clip, self.base_visual_proj]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.classifier = nn.Linear(self.base_visual_proj.out_features, num_classes)

    def forward(self, x: Dict) -> Tensor:
        """Pass forward.

        Args:
            x: Dictionary of pixel values with shape (BS, F, C, H, W).

        Returns:
            prediction per video (BS, num_classes)
        """
        x["pixel_values"] = x["pixel_values"].squeeze()
        z = self.base_visual_clip(**x).pooler_output
        z = self.base_visual_proj(z)
        if self.output_embed:
            return z
        z = self.classifier(z)
        # Do not apply softmax if there is a single class!!
        return nn.functional.log_softmax(z, dim=1) if self.classifier.out_features > 1 else torch.tanh(z)


class CLIP(nn.Module):
    def __init__(self, output_embed: bool = False):
        super(CLIP, self).__init__()
        self.vision = CustomVisualCLIP(num_classes=0, finetune=False)
        self.textual = CustomTextualCLIP(num_classes=0, finetune=False, multisentence=False)
        self.output_embed = output_embed

        # https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPConfig.logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, x: Dict) -> torch.Tensor:
        z_vision = self.vision(x[0])
        z_textual = self.textual(x[1])

        # normalized features
        image_embeds = z_vision / z_vision.norm(p=2, dim=-1, keepdim=True)
        text_embeds = z_textual / z_textual.norm(p=2, dim=-1, keepdim=True)
        if self.output_embed:
            return torch.stack((image_embeds, text_embeds), dim=0)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return torch.stack([logits_per_image, logits_per_text], dim=0)


def get_visual_embeddings(
    model: nn.Module, processor: CLIPProcessor, image_paths: list, device: torch.device
) -> numpy.ndarray:
    """Get the embeddings of a list of images.

    Args:
        model: CLIP model.
        image_paths: List of paths to images.
        device: Device to use.

    Returns:
        Embeddings of the images.

    """
    from PIL import Image
    from tqdm import tqdm

    model.eval()
    model.to(device)
    embeddings = numpy.empty((0, 512))
    # Automatically batch images for speed
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch = image_paths[i: i + batch_size]
            images = [Image.open(path) for path in batch]
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            vis_emb = model.vision(img_inputs) if isinstance(model, CLIP) else model.get_image_features(**img_inputs)
            embeddings = numpy.concatenate((embeddings, vis_emb.detach().cpu().numpy()))
    return embeddings


def get_textual_embeddings(
    model: nn.Module, processor: CLIPProcessor, texts: list, device: torch.device
) -> numpy.ndarray:
    """Get the embeddings of a list of texts.

    Args:
        model: CLIP model.
        texts: List of texts.
        device: Device to use.

    Returns:
        Embeddings of the texts.

    """
    from tqdm import tqdm

    model.eval()
    model.to(device)
    embeddings = numpy.empty((0, 512))
    # Automatically batch texts for speed
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i: i + batch_size]
            text_inputs = processor(text=batch, return_tensors="pt", padding=True).to(device)
            text_emb = model.textual(text_inputs) if isinstance(model, CLIP) else model.get_text_features(**text_inputs)
            embeddings = numpy.concatenate((embeddings, text_emb.detach().cpu().numpy()))
    return embeddings
