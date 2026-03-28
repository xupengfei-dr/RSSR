# coding=utf-8
# Copyright 2022 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BLIP model."""
import importlib
import os
import warnings

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import matplotlib
from PIL import Image

from ... import SiglipModel

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.functional import normalize


from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_blip_text import BlipTextLMHeadModel,BlipTextModel

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/blip-vqa-base"

BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip-vqa-base",
    "Salesforce/blip-vqa-capfilt-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-flickr",
    "Salesforce/blip-itm-large-flickr",
    # See all BLIP models at https://huggingface.co/models?filter=blip
]


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->blip
def blip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class BlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def decoder_logits(self):
        warnings.warn(
            "`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the `logits` attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.logits


@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    """

    loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state_vero:Optional[torch.FloatTensor] = None
    last_hidden_state_encoder:Optional[torch.FloatTensor] = None
    input_embedding_attention_mask:Optional[torch.IntTensor] = None



@dataclass
class BlipImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        itm_score (`torch.FloatTensor`):
            The image-text similarity scores.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        vision_pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        question_embeds (`torch.FloatTensor`):
            The question embeddings obtained by the text projection layer.
    """

    itm_score: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_pooler_output: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    question_embeds: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BlipOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class BlipVisionEmbeddings(nn.Module):
    def __init__(self, config: BlipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        # img size = 384 patch = 16 num_patch = 576
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # self.num_patches = (512 // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]

        target_dtype = self.patch_embedding.weight.dtype

        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # pixel = 64,3,512,512  patch_embeds = 64,768,32,32   ->patch_embeds = 64,768,1024

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)  #class model = 64,1,768

        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)   # embeddings = 64,1025,768

        if False:
            position_embedding2 = nn.Parameter(torch.randn(1, 1025, 768))
            position_embedding2 = position_embedding2.to(self.position_embedding.device)
            embeddings = embeddings + position_embedding2[:, : embeddings.size(1), :].to(target_dtype)
        else:
            embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->Blip
class BlipTextEmbeddings(nn.Module):
    def __init__(self, config: BlipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class BlipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = (
            self.qkv(hidden_states)
            .reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Blip
class BlipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BlipEncoderLayer(nn.Module):
    def __init__(self, config: BlipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = BlipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = BlipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BlipConfig
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, BlipVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(
                module.position_embedding,
                mean=0.0,
                std=factor,
            )

            nn.init.trunc_normal_(
                module.class_embedding,
                mean=0.0,
                std=factor,
            )

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BlipEncoder):
            module.gradient_checkpointing = value


BLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BlipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`BlipEncoderLayer`].

    Args:
        config (`BlipConfig`):
            The corresponding vision configuration for the `BlipEncoder`.
    """

    def __init__(self, config: BlipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([BlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BlipVisionModel(BlipPreTrainedModel):
    main_input_name = "pixel_values"
    config_class = BlipVisionConfig

    def __init__(self, config: BlipVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = BlipVisionEmbeddings(config)
        self.encoder = BlipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings



# TODO:TEST  该模型流程为：嵌入图片和文本向量，使用其向量L2归一化并计算余弦相似度返回 ，没有进入编码层

@add_start_docstrings(BLIP_START_DOCSTRING)
class BlipModel(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = BlipTextModel(text_config)
        self.vision_model = BlipVisionModel(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`BlipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, BlipModel

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`BlipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipModel

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features


    '''
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
    '''

    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipOutput, config_class=BlipConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipModel

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use BLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 视觉嵌入输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 文本向量嵌入输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 视觉向量
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        #文本向量
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features   单位归一化，以求余弦相似度
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits   计算图文相似度采用余弦
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = blip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return BlipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


@add_start_docstrings(
    """
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """,
    BLIP_START_DOCSTRING,
)

# 图文生成
class BlipForConditionalGeneration(BlipPreTrainedModel):
    config_class = BlipConfig
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]
    main_input_name = "pixel_values"

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipForConditionalGenerationModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model(**inputs)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return BlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration

        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats sleeping on a couch
        ```
        """

        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs


'''
        self.attn_adapter = RSAdapter(config.hidden_size, D_dim=192, skip_connect=False)
        self.mlp_adapter = RSAdapter(config.hidden_size, D_dim=192, skip_connect=False)
        self.attn_adapter_scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.mlp_adapter_scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
'''
# RSAdapter
class RSAdapter(nn.Module):
    def __init__(self, D_features, D_dim=192, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_dim)
        self.D_fc2 = nn.Linear(D_dim, D_features)
        self.weight_1, self.bias_1 = self.init_LT(D_dim)
        self.weight_2, self.bias_2 = self.init_LT(D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.LT(xs, self.weight_1, self.bias_1)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = self.LT(xs, self.weight_2, self.bias_2)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs

        return x

    def LT(self, x, weight, bias):
        return x * weight + bias

    def init_LT(self, dim):
        weight = nn.Parameter(torch.ones(dim))
        bias = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(weight, mean=1, std=.02)
        nn.init.normal_(bias, std=.02)

        return weight, bias




@add_start_docstrings(
    """
    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.
    """,
    BLIP_START_DOCSTRING,
)

# todo： 使用blipfor vqa

class BlipForQuestionAnswering(BlipPreTrainedModel):
    config_class = BlipConfig
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]

    def __init__(self, config: BlipConfig):
        super().__init__(config)
        vision_config = config.vision_config
        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id
        # insert
        # self.encoder = BlipEncoder(config.vision_config)
        # self.vision_model_plus = SiglipModel.from_pretrained("")
        # self.vision_model_plus = SiglipModel.from_pretrained("/home/pengfei/siglip2-large-patch16-256").vision_model


        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding
    '''
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        answer = batch["answer"]
    '''
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForQuestionAnswering

        >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # training
        >>> text = "How many cats are in the picture?"
        >>> label = "2"
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> labels = processor(text=label, return_tensors="pt").input_ids

        >>> inputs["labels"] = labels
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # inference
        >>> text = "How many cats are in the picture?"
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```"""
        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either `decoder_input_ids` or `labels` should be passed when calling `forward` with"
                " `BlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you"
                " are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`"
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 输入通常是预处理后的图像张量，输出是图像的深层语义表示（编码特征）或全局特征（池化特征），可供下游任务进一步使用。
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(attention_mask.device)
        # input_embedding_attention_mask = torch.cat([attention_mask, image_attention_mask], dim=1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_ids.shape)

        # 进行文本编码， 多模态交互 ， 特征输出
        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        # cross_attentions = question_embeds['cross_attentions']
        #
        # import cv2
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from PIL import Image
        # import os

        # # === 配置 ===
        # image_path = "/home/pengfei/rsvqa/data/RSVQA_LR/Image/Images_LR/245.tif"
        # image_id = "245"  # 232- 0-100
        # output_base_dir = "/home/pengfei/mamba/assets"  # 保存文件的基础目录
        # num_questions = 120  # 要处理的问题（热力图）数量
        # start_batch_idx = 1300  # 第一个问题对应的 batch_idx
        # layer_idx = 11  # 选择的层
        # head_idx = 7  # 选择的头
        # alpha = 0.6  # 叠加透明度
        # save_format = 'svg'  # 保存格式
        #
        # # --- 新增：模糊配置 ---
        # blur_ksize = 31  # 高斯核大小 - 必须是奇数。值越大越模糊。可以试试 15, 21, 31, 51 等。
        #
        # # --- 确保 blur_ksize 是奇数 ---
        # if blur_ksize % 2 == 0:
        #     blur_ksize += 1
        #     print(f"   已将 blur_ksize 调整为奇数: {blur_ksize}")
        # # ------------------------------
        #
        # # --- 在循环外执行一次的操作 ---
        #
        # # === 1. 读取原始 TIFF 图像 ===
        # try:
        #     original_image_pil = Image.open(image_path)
        #     original_image = np.array(original_image_pil)
        #
        #     # 如果图像是灰度图像，转换为 RGB
        #     if len(original_image.shape) == 2:
        #         original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        #     elif original_image.shape[2] == 4:  # 处理 RGBA
        #         original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        #     elif original_image.shape[2] == 3:
        #         print("图像已经是 3 通道。")
        #     else:
        #         raise ValueError(f"不支持的图像形状: {original_image.shape}")
        #
        #     # 确保数据类型是 uint8 以便叠加
        #     if original_image.dtype != np.uint8:
        #         print(f"正在将图像数据类型从 {original_image.dtype} 转换为 uint8。")
        #         if np.max(original_image) > 255:
        #             original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #         else:
        #             original_image = original_image.astype(np.uint8)
        #
        #     image_h, image_w, _ = original_image.shape  # 获取图像尺寸
        #
        # except FileNotFoundError:
        #     print(f"错误: 图像文件未找到 {image_path}")
        #     exit()
        # except Exception as e:
        #     print(f"错误: 加载或预处理图像时出错: {e}")
        #     exit()
        #
        # # 创建输出目录 (如果不存在)
        # # 在输出目录名中加入模糊信息，方便区分不同模糊程度的结果
        # output_dir = os.path.join(output_base_dir, f"heatmaps_11t200_mh_notzengqiang_{image_id}_blurred{blur_ksize}")
        # os.makedirs(output_dir, exist_ok=True)
        # print(f"热力图将保存到: {output_dir}")
        #
        # # --- 循环处理每个问题 ---
        # for i in range(num_questions):
        #     current_batch_idx = start_batch_idx + i
        #     current_question_index = i  # 从 0 开始计数
        #
        #     print(f"--- 处理问题 {current_question_index} (Batch Index: {current_batch_idx}) ---")
        #
        #     try:
        #         # === 2. 获取当前问题的 cross attention 数据 ===
        #         # 检查 cross_attentions 是否已定义
        #         if 'cross_attentions' not in locals() and 'cross_attentions' not in globals():
        #             # --- 用于测试的伪造数据 (在你的实际代码中移除这部分) ---
        #             # 请用加载你实际 cross_attentions 数据的代码替换这里
        #             print("   警告: 'cross_attentions' 未定义，正在使用随机测试数据。")
        #             # 示例形状: (层数, 批次大小, 头数, 查询长度, 键长度)
        #             # 假设 键长度 = patch数量 + 1 (CLS)
        #             # 假设 patch数量 = 16*16 = 256. 键长度 = 257
        #             # 假设 查询长度 = 50 (示例)
        #             # batch_idx 最大到 150+200=350，所以批次大小需要 >= 350
        #             # 假设 12 层, 批次大小 400, 8 个头
        #             fake_batch_size = 400
        #             fake_num_patches = 256
        #             fake_key_len = fake_num_patches + 1
        #             fake_query_len = 50
        #             cross_attentions = [
        #                 np.random.rand(fake_batch_size, 8, fake_query_len, fake_key_len).astype(np.float32) for _ in
        #                 range(12)]
        #             # -----------------------------------------------------------
        #             # raise NameError("变量 'cross_attentions' 未定义。请确保在循环开始前加载它。") # 实际代码中保留这行检查
        #
        #         # 尝试访问数据，处理可能的索引错误
        #         try:
        #             # 假设 cross_attentions 是包含 numpy 数组的列表或其他可索引结构
        #             attention_map_raw = cross_attentions[layer_idx][current_batch_idx, head_idx, :, :]
        #         except IndexError:
        #             print(f"   错误: 无法访问 cross_attentions[{layer_idx}][{current_batch_idx}, {head_idx}, :, :]。")
        #             print(f"   请检查 cross_attentions 的维度和 batch_idx {current_batch_idx} 是否有效。")
        #             continue  # 跳过这个问题的处理
        #
        #         # === 3. 归一化处理 ===
        #         attention_map = np.mean(np.array(attention_map_raw), axis=0)  # 确保是 NumPy 数组并求平均
        #         attention_map = attention_map[1:]  # 去掉 CLS token
        #
        #         num_patches = attention_map.shape[0]
        #         patch_grid_size = int(np.sqrt(num_patches))
        #         if patch_grid_size * patch_grid_size != num_patches:
        #             print(f"   警告: Token 数量 ({num_patches}) 不能完美开方成方形网格。跳过问题 {current_question_index}。")
        #             continue
        #
        #         attention_map = attention_map.reshape(patch_grid_size, patch_grid_size)
        #
        #         map_min, map_max = np.min(attention_map), np.max(attention_map)
        #         if map_max - map_min < 1e-6:
        #             attention_map_normalized = np.zeros_like(attention_map)
        #             print(f"   警告: 问题 {current_question_index} 的注意力图值域过小，生成全零热力图。")
        #         else:
        #             attention_map_normalized = (attention_map - map_min) / (map_max - map_min)
        #
        #         # === 4. 调整尺寸，使其与原图匹配 ===
        #         attention_map_resized = cv2.resize(attention_map_normalized, (image_w, image_h),
        #                                            interpolation=cv2.INTER_LINEAR)
        #
        #         # === 新增步骤: 4.5 应用高斯模糊 ===
        #         # 对调整尺寸后的、归一化的注意力图应用模糊
        #         blurred_attention_map = cv2.GaussianBlur(attention_map_resized, (blur_ksize, blur_ksize), 0)
        #         # ------------------------------------
        #
        #         # === 5. 生成颜色映射（伪彩色） ===
        #         # 使用模糊后的图来生成热力图
        #         heatmap = cv2.applyColorMap(np.uint8(255 * blurred_attention_map),
        #                                     cv2.COLORMAP_JET)  # 使用 blurred_attention_map
        #         heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转为 RGB
        #
        #         # === 6. 将热力图叠加到原图上 ===
        #         # 如果热力图因为模糊显得太强，可以稍微增加原图的权重
        #         overlayed_image = cv2.addWeighted(original_image, 0.7, heatmap_rgb, alpha, 0)  # 调整了原图权重
        #
        #         # === 7. 保存结果 ===
        #         plt.figure(figsize=(6, 6))
        #         plt.imshow(overlayed_image)
        #         # 在标题中加入模糊信息
        #         plt.title(f"Image {image_id} - Q {current_question_index:03d} Heatmap (Blur {blur_ksize})", fontsize=9)
        #         plt.axis("off")
        #
        #         # 在保存文件名中加入模糊信息
        #         save_filename = f"heatmap_img{image_id}__q{current_question_index:03d}_blur{blur_ksize}.{save_format}"
        #         save_path = os.path.join(output_dir, save_filename)
        #
        #         plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        #         plt.close()  # 关闭图形，释放内存
        #
        #     except Exception as e:
        #         print(f"   处理问题 {current_question_index} (Batch Index: {current_batch_idx}) 时发生错误: {e}")
        #         # 如果需要详细调试，可以取消注释下面这行来打印错误追踪信息
        #         # import traceback
        #         # traceback.print_exc()
        #         plt.close()  # 确保即使出错也关闭可能已创建的图形
        #         continue  # 继续处理下一个问题
        #
        # print(f"-------------------------------- 所有问题处理完成 (模糊核大小={blur_ksize}) -----------------------")
        # # 除非你明确需要，否则移除最后的 exit()
        #todo:--------------------------------------------------
        #
        # # === 配置 ===
        # image_path = "/home/pengfei/rsvqa/data/RSVQA_LR/Image/Images_LR/244.tif"
        # image_id = "244"  #232- 0-100
        # output_base_dir = "/home/pengfei/mamba/assets"  # 保存文件的基础目录
        # num_questions = 100 # 要处理的问题（热力图）数量
        # start_batch_idx = 1200  # 第一个问题对应的 batch_idx
        # layer_idx = 11  # 选择的层
        # head_idx = 7  # 选择的头
        # alpha = 0.6  # 叠加透明度
        # save_format = 'svg'  # 保存格式
        #
        # # --- 在循环外执行一次的操作 ---
        #
        # # === 1. 读取原始 TIFF 图像 ===
        # try:
        #     original_image_pil = Image.open(image_path)
        #     original_image = np.array(original_image_pil)
        #
        #     # 如果图像是灰度图像，转换为 RGB
        #     if len(original_image.shape) == 2:
        #         original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        #     elif original_image.shape[2] == 4:  # 处理 RGBA
        #         original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        #     elif original_image.shape[2] == 3:
        #         print("Image is already 3 channels.")
        #     else:
        #         raise ValueError(f"Unsupported image shape: {original_image.shape}")
        #
        #     # 确保数据类型是 uint8 以便叠加
        #     if original_image.dtype != np.uint8:
        #         print(f"Converting image dtype from {original_image.dtype} to uint8.")
        #         if np.max(original_image) > 255:
        #             original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #         else:
        #             original_image = original_image.astype(np.uint8)
        #
        #     image_h, image_w, _ = original_image.shape  # 获取图像尺寸
        #
        # except FileNotFoundError:
        #     print(f"错误: 图像文件未找到 {image_path}")
        #     exit()
        # except Exception as e:
        #     print(f"错误: 加载或预处理图像时出错: {e}")
        #     exit()
        #
        # # 创建输出目录 (如果不存在)
        # output_dir = os.path.join(output_base_dir, f"heatmaps_11t200_notzengqiang2_{image_id}")
        # os.makedirs(output_dir, exist_ok=True)
        # print(f"热力图将保存到: {output_dir}")
        #
        # # --- 循环处理每个问题 ---
        # for i in range(num_questions):
        #     current_batch_idx = start_batch_idx + i
        #     current_question_index = i  # 从 0 开始计数
        #
        #     print(f"--- 处理问题 {current_question_index} (Batch Index: {current_batch_idx}) ---")
        #
        #     try:
        #         # === 2. 获取当前问题的 cross attention 数据 ===
        #         # 检查 cross_attentions 是否已定义
        #         if 'cross_attentions' not in locals() and 'cross_attentions' not in globals():
        #             raise NameError("变量 'cross_attentions' 未定义。请确保在循环开始前加载它。")
        #
        #         # 尝试访问数据，处理可能的索引错误
        #         try:
        #             attention_map_raw = cross_attentions[layer_idx][current_batch_idx, head_idx, :, :]
        #         except IndexError:
        #             print(f"   错误: 无法访问 cross_attentions[{layer_idx}][{current_batch_idx}, {head_idx}, :, :]。")
        #             print(f"   请检查 cross_attentions 的维度和 batch_idx {current_batch_idx} 是否有效。")
        #             continue  # 跳过这个问题的处理
        #
        #         # 如果是 PyTorch Tensor，移到 CPU 并 detach
        #         if hasattr(attention_map_raw, 'cpu'):
        #             attention_map_raw = attention_map_raw.cpu()
        #         if hasattr(attention_map_raw, 'detach'):
        #             attention_map_raw = attention_map_raw.detach()
        #
        #         # === 3. 归一化处理 ===
        #         # 确保对 dim=0 (query dimension) 求平均
        #         if hasattr(attention_map_raw, 'mean'):  # PyTorch Tensor or NumPy array
        #             attention_map = attention_map_raw.mean(dim=0 if hasattr(attention_map_raw, 'dim') else 0)
        #         else:  # Fallback for other array types if needed
        #             attention_map = np.mean(np.array(attention_map_raw), axis=0)
        #
        #         attention_map = attention_map[1:]  # 去掉 CLS token
        #
        #         # 假设 token 是 16x16 网格
        #         num_patches = attention_map.shape[0]
        #         patch_grid_size = int(np.sqrt(num_patches))
        #         if patch_grid_size * patch_grid_size != num_patches:
        #             print(f"   警告: Token 数量 ({num_patches}) 不能完美开方成方形网格。跳过问题 {current_question_index}。")
        #             continue
        #
        #         attention_map = attention_map.reshape(patch_grid_size, patch_grid_size)
        #
        #         # 转换为 NumPy (如果需要)
        #         if not isinstance(attention_map, np.ndarray):
        #             attention_map = attention_map.numpy()
        #
        #         # 归一化到 [0, 1]
        #         map_min, map_max = np.min(attention_map), np.max(attention_map)
        #         if map_max - map_min < 1e-6:
        #             attention_map_normalized = np.zeros_like(attention_map)
        #             print(f"   警告: 问题 {current_question_index} 的注意力图值域过小，生成全零热力图。")
        #         else:
        #             attention_map_normalized = (attention_map - map_min) / (map_max - map_min)
        #
        #         # === 4. 调整尺寸，使其与原图匹配 ===
        #         attention_map_resized = cv2.resize(attention_map_normalized, (image_w, image_h),
        #                                            interpolation=cv2.INTER_LINEAR)
        #
        #         # === 5. 生成颜色映射（伪彩色） ===
        #         heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
        #         heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转为 RGB
        #
        #         # === 6. 将热力图叠加到原图上 ===
        #         overlayed_image = cv2.addWeighted(original_image, 0.8, heatmap_rgb, alpha, 0)
        #
        #         # === 7. 保存结果 (不再显示原图，只保存叠加后的热力图) ===
        #         plt.figure(figsize=(6, 6))  # 创建一个适合保存的图形尺寸
        #         plt.imshow(overlayed_image)
        #         plt.title(f"Image {image_id} - Q {current_question_index:03d} Heatmap", fontsize=10)  # 添加标题区分
        #         plt.axis("off")
        #
        #         # 构建动态保存路径
        #         save_filename = f"heatmap_img{image_id}__q{current_question_index:03d}.{save_format}"
        #         save_path = os.path.join(output_dir, save_filename)
        #
        #         plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        #         plt.close()  # 关闭图形，释放内存，非常重要！
        #
        #         # print(f"   已保存: {save_path}")
        #
        #     except Exception as e:
        #         print(f"   处理问题 {current_question_index} (Batch Index: {current_batch_idx}) 时发生错误: {e}")
        #         # 确保即使出错也关闭可能已创建的图形
        #         plt.close()
        #         continue  # 继续处理下一个问题
        #
        # print("-------------------------------- 所有问题处理完成 -----------------------")
        # exit()
        #todo:--------------------------------------------------
        #todo:----------------------------------------------------------
        #
        # === 1. 读取原始 TIFF 图像 ===
        # image_path = "/home/pengfei/rsvqa/data/RSVQA_LR/Image/Images_LR/238.tif"  # 请替换为你的 TIFF 图像路径
        # original_image = Image.open(image_path)  # 使用 PIL 读取 TIFF 图像
        # original_image = np.array(original_image)  # 转换为 NumPy 数组
        #
        # # 如果图像是灰度图像，转换为 RGB
        # if len(original_image.shape) == 2:  # 只有一个通道（灰度图像）
        #     original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)  # 转为 RGB 图像
        # elif original_image.shape[2] == 3:  # 已经是 RGB 或 BGR 图像
        #     print("Image is already in RGB format.")
        # else:
        #     print("Unknown image format.")
        #
        # image_h, image_w, _ = original_image.shape  # 获取图像尺寸
        #
        # # === 2. 获取 cross attention 数据 ===
        # # 假设 cross_attentions 是一个形状为 [12, 64, 12, 40, 257] 的数组，包含多个层、多个头的注意力
        # layer_idx = 6  # 选择某一层 (这里是第 12 层)
        # batch_idx = 630 # 选择 batch 里的第 1 个样本
        # head_idx = 6  # 选择第 1 个注意力头
        # attention_map = cross_attentions[layer_idx][batch_idx, head_idx, :, :]  # shape: [40, 257]
        #
        # # === 3. 归一化处理 ===
        # attention_map = attention_map.mean(dim=0)  # 对 40 个 Query 取平均，得到 shape: [257]
        # attention_map = attention_map[1:]  # 去掉 CLS token
        # attention_map = attention_map.reshape(16, 16)  # 假设 token 是 16x16 格式
        # attention_map = attention_map.cpu().numpy()  # 转换为 NumPy 格式
        # attention_map = (attention_map - np.min(attention_map)) / (
        #         np.max(attention_map) - np.min(attention_map))  # 归一化到 [0,1]
        #
        # # === 4. 调整尺寸，使其与原图匹配 ===
        # attention_map_resized = cv2.resize(attention_map, (image_w, image_h), interpolation=cv2.INTER_LINEAR)  # 调整到原图大小
        #
        # # === 5. 生成颜色映射（伪彩色） ===
        # heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)  # 使用 JET 颜色映射
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        #
        # # === 6. 将热力图叠加到原图上 ===
        # alpha = 0.8  # 透明度
        # overlayed_image = cv2.addWeighted(original_image, 0.9, heatmap, alpha, 0)
        #
        # # === 7. 显示结果 ===
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_image)
        # plt.title("Original tif",fontsize= 15)
        # plt.axis("off")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(overlayed_image)
        # # plt.title(f"Attention Heatmap (Layer {layer_idx + 1}, Head {head_idx + 1})")
        # plt.title("Attention Heatmap",fontsize=15)
        # plt.axis("off")
        #
        # plt.savefig("/home/pengfei/mamba/assets/relitu_359234_238_02.svg", bbox_inches='tight', pad_inches=0,
        #             dpi=300)
        #
        # print("-------------------------------- save -----------------------")
        # exit()
        last_hidden_state_vero = question_embeds['last_hidden_state']

        # result_encoder = self.encoder(inputs_embeds=last_hidden_state_vero)



        # todo：在这将question_embeds插入encoder解码层 ：
        '''
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        '''




        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            # 这里labels已经右移了，不用再右移了
            decoder_input_ids = labels
        # todo:查看融合后的向量的last_hidden_state
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        # TODO:
        # # 在文本编码器输出后应用RSAdapter
        # attn_adapter_output = self.attn_adapter(question_embeds)
        # question_embeds = question_embeds + attn_adapter_output * self.attn_adapter_scale
        #
        # # 在文本解码器前应用RSAdapter
        # mlp_adapter_output = self.mlp_adapter(question_embeds)
        # question_embeds = question_embeds + mlp_adapter_output * self.mlp_adapter_scale

        # todo：解码层。。。
        answer_output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        if labels is not None:
            decoder_loss = answer_output.loss.mean() if return_dict else answer_output[0].mean()
        else:
            decoder_loss = None

        if not return_dict:
            outputs = (decoder_loss, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)
        # todo：这个last_hidden_state 是vision_outputs层最后一层的状态
        return BlipTextVisionModelOutput(
            loss=decoder_loss,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            last_hidden_state_vero =last_hidden_state_vero,
            input_embedding_attention_mask=extended_attention_mask,
            # last_hidden_state_encoder =result_encoder['last_hidden_state'],

            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*):
                The sequence used as a prompt for the generation.
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            **generate_kwargs:
                Additional arguments passed to the *generate* function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForQuestionAnswering

        >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```
        """
        # todo:图像嵌入向量
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        # todo:文本向量为与图像的嵌入向量相互作用生成的
        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )
        question_embeds = question_outputs[0]

        # question_embeds = question_embeds + self.attn_adapter(question_embeds) * self.attn_adapter_scale
        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)
        bos_ids = torch.full(
            (question_embeds.size(0), 1), fill_value=self.decoder_start_token_id, device=question_embeds.device
        )
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        return outputs


@add_start_docstrings(
    """
    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
    the image.
    """,
    BLIP_START_DOCSTRING,
)

class BlipForImageTextRetrieval(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # vision projection layer
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForImageTextRetrieval

        >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        if use_itm_head:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            output = self.itm_head(question_embeds[:, 0, :])
        else:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            output = image_feat @ text_feat.t()

        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple(output for output in outputs if output is not None)

        return BlipImageTextMatchingModelOutput(
            itm_score=output,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            question_embeds=question_embeds,
        )
