from typing import Optional
import numpy as np
import torch
from torch import nn
from transformers.image_processing_utils import select_best_resolution

select_best_resolution
# device = torch.device("cuda:0")
# from transformers import AutoModel, AutoConfig, CLIPConfig, PretrainedConfig  # 导入需要的类
from  transformers import CLIPConfig, PretrainedConfig, AutoModel, AutoConfig, AutoModelForCausalLM, \
    AutoTokenizer, CLIPProcessor, AutoProcessor, CLIPVisionModel, Qwen2ForCausalLM, Qwen2Model


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise ValueError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding: current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding: current_width - padding]

    return unpadded_tensor


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`Union[torch.LongTensor, np.ndarray, Tuple[int, int]):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise ValueError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


class MyVLQConfig(PretrainedConfig):
    def __init__(
            self,
            vision_path,
            text_path,
            fusion_hidden_dim=512,
            num_classes=10,
            pretrained_vision_name_or_path=None,  # 新增：预训练视觉模型的名称或路径
            **kwargs
    ):

        super().__init__(**kwargs)
        # 如果传入字典类型, 用CLIPConfig进行配置
        self.vision_model = CLIPVisionModel.from_pretrained(vision_path)
        self.vision_config = self.vision_model.vision_model.config
        self.image_token_index = 151665
        self.ignore_index = -100
        self.vocab_size = 32064
        self.torch_dtype = "float16"
        self.text_model = Qwen2Model.from_pretrained(text_path)
        self.text_config = self.text_model.config
        self.vision_feature_layer = -2
        self.fusion_hidden_dim = fusion_hidden_dim  # 维度信息
        self.num_classes = num_classes  # 分类信息
        self.pretrained_vision_name_or_path = pretrained_vision_name_or_path
        self.image_grid_pinpoints = [
                                        [
                                            336,
                                            672
                                        ],
                                        [
                                            672,
                                            336
                                        ],
                                        [
                                            672,
                                            672
                                        ],
                                        [
                                            1008,
                                            336
                                        ],
                                        [
                                            336,
                                            1008
                                        ]
                                    ]

    def to_dict(self):  # 转成dict格式
        output = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
            else:
                output[key] = value

        return output

    def __repr__(self) -> str:  # 方便print
        return str(self.to_dict())  #


class MultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_dim, text_hidden_dim):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_dim, text_hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_dim, text_hidden_dim)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CoAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=1536, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 图像到文本的注意力
        self.img2text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 文本到图像的注意力
        self.text2img_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 残差连接的Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features, text_features):
        """
        输入:
        img_features:   [batch_size, 576, 1536]
        text_features: [batch_size, 40, 1536]

        输出:
        fused_features: [batch_size, 576+40, 1536]
        """
        # --------------------------
        # 第一阶段：图像到文本的注意力
        # --------------------------
        # 使用图像作为Query，文本作为Key/Value
        img_attn_out, _ = self.img2text_attn(
            query=img_features,
            key=text_features,
            value=text_features
        )
        img_attn_out = self.norm1(img_features + self.dropout(img_attn_out))

        # --------------------------
        # 第二阶段：文本到图像的注意力
        # --------------------------
        # 使用文本作为Query，图像作为Key/Value
        text_attn_out, _ = self.text2img_attn(
            query=text_features,
            key=img_attn_out,
            value=img_attn_out
        )
        text_attn_out = self.norm2(text_features + self.dropout(text_attn_out))

        # --------------------------
        # 第三阶段：特征拼接与融合
        # --------------------------
        # 拼接两种模态特征
        combined = torch.cat([img_attn_out, text_attn_out], dim=1)

        # 前馈网络增强
        fused_features = self.feed_forward(combined)
        fused_features = self.norm3(combined + self.dropout(fused_features))

        return fused_features


class VLQ(nn.Module):
    def __init__(self, config: MyVLQConfig):
        super().__init__()
        self.config = config
        self.vision_tower = config.vision_model.vision_model
        self.language_model = config.text_model
        self.conattention = CoAttentionFusion(hidden_dim=config.text_config.hidden_size)
        self.multi_modal_projector = MultiModalProjector(
            vision_hidden_dim=config.vision_config.hidden_size,
            text_hidden_dim=config.text_config.hidden_size
            # config.text_config.hidden_size,  #  text_config 如果为None,就随意传入一个int
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.init_weights()

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def pack_image_features(self, image_features, image_sizes, image_newline=None):

        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02  #
        )

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, inputs_embeds=None,
                image_sizes: Optional[torch.LongTensor] = None, ):


        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[self.config.vision_feature_layer]  # 1,577,1024
        image_features = image_features[:, 1:]  # [CLS] token  1,576,1024


        # todo :更新
        print("------------- get feature----------------------")
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
            inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

        # 3. 特征对齐 ，将视觉的大小投影到文本的长度上
        image_features = self.multi_modal_projector(image_features)
        inputs_embeds = inputs_embeds.to(image_features.dtype)
        print(inputs_embeds.shape)
        print(image_features[0].shape)



        print("------------------ merge-----------------")

        inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )

        print("++++++++++++++ result++++++++++++++")
        print(inputs_embeds.shape)
        print(attention_mask.shape)
        print(position_ids.shape)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False,
        )

        logits = outputs[0]

        print(outputs)

        print("demo passed")

        exit()

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return logits, loss

#
# if __name__ == '__main__':
#     # qwen_model_path = "/home/pengfei/Qwen1.5-4B-Chat"
#     qwen_model_path = "/home/pengfei/Qwen2.5-1.5B-Instruct"
#
#     # --- 虚拟输入 ---
#     batch_size = 2
#     input_ids_question = "<image> What is the number of residential buildings?"
#     llm_model_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
#     inputs = llm_model_tokenizer(input_ids_question, return_tensors="pt", max_length=40, padding="max_length")
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)
#     labels = torch.randint(0, 9, (batch_size,)).to(device)  #
#     print(input_ids)
#     print(attention_mask)
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw)
#     clip_vit_model_Path = "/home/pengfei/clip-vit-large-patch14-336"
#     processor = AutoProcessor.from_pretrained(clip_vit_model_Path)
#     pixel_values = processor.image_processor(images=image, return_tensors="pt")
#     pixel_values2 = pixel_values['pixel_values'].to(device)
#     config = MyVLQConfig(vision_path=clip_vit_model_Path, text_path=qwen_model_path)
#     model = VLQ(config)
#     model.to(device)
#     for param in model.parameters():
#         param.requires_grad = False
#     print(model)  # 输出模型结构
#     # --- 前向传播 ---
#     logits, loss = model(pixel_values2, input_ids, attention_mask)
#     print("Logits:", logits.shape)
#     print("loss", loss)
