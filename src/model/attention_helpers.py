# 文件: attention_helpers.py

import torch
import torch.nn as nn
import math
import warnings
from typing import Optional


# ==========================================================
# 1. 你的配置类
# ==========================================================
class FlashAttConfig:
    """一个统一的配置类，用于初始化所有注意力模块"""

    # head_dim 是计算出来的，不是预设的
    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    rope_theta = 500000.0
    hidden_size = 768
    num_attention_heads = 16  # 这是 MHSA 和 GQA 中 Q 的头数
    num_key_value_heads = 8  # GQA 中 K 和 V 的头数
    attention_dropout = 0.1
    max_position_embeddings = 2048
    rope_scaling = None


# ==========================================================
# 2. RoPE 辅助函数和类
# (从 Hugging Face Transformers 简化而来，用于独立运行)
# ==========================================================

def rotate_half(x):
    """旋转输入张量的一半维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """将旋转位置编码应用到 Q 和 K 张量上"""
    cos = cos.unsqueeze(1)  # 形状变为 (B, 1, N, hD/2*2)
    sin = sin.unsqueeze(1)  # 形状变为 (B, 1, N, hD/2*2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DummyLogger:
    def warning_once(self, msg):
        warnings.warn(msg, UserWarning)


logger = DummyLogger()


def _create_default_rope_init_function(config, device, **kwargs):
    inv_freq = 1.0 / (config.rope_theta ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.int64).float().to(device) / config.head_dim))
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS = {"default": _create_default_rope_init_function}


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: FlashAttConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)