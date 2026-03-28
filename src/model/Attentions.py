# File: final_attention_modules.py
# (Make sure to import from attention_helpers.py)

import torch
import torch.nn as nn
import math
from typing import Optional

# Import everything from our helper file

# Try to import flash-attn
from src.flash_attention2_test import FlashAttConfig, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    print("Warning: flash-attn library not found. FlashAttention modules will not be available.")
    FLASH_ATTN_AVAILABLE = False

# ==========================================================
# 模块 1: MHSA + AbsPE forward(feature,attentionmask)
# ==========================================================
class MHSA_AbsPE_Internal(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) with internal Absolute Positional Encoding (AbsPE).
    """
    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # 投影
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

        # 可学习的绝对位置编码
        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # 加入位置编码
        x = x + self.pos_embed(self.position_ids[:, :N])

        # 投影并切分多头
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o_proj(out)

# ==========================================================
# 模块 2: MHSA + RoPE (PyTorch Standard Implementation)
# ==========================================================
class MHSA_RoPE_Internal(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) with Rotary Positional Encoding (RoPE).
    """
    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # 投影
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

        # 固定 RoPE
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # 投影并切分多头
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        position_ids = torch.arange(N, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o_proj(out)


# ==========================================================
# 模块 3: GQA + AbsPE (PyTorch Standard Implementation)
# ==========================================================
class GQA_AbsPE_Internal(nn.Module):
    """
    Grouped-Query Attention (GQA) with internal Absolute Positional Encoding (AbsPE).
    Uses standard PyTorch backend for attention calculation.
    """

    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = x + self.pos_embed(self.position_ids[:, :N])

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o_proj(out)


# ==========================================================
# 模块 4: GQA + RoPE (PyTorch Standard Implementation)
# ==========================================================
class GQA_RoPE_Internal(nn.Module):
    """
    Grouped-Query Attention (GQA) with internal Llama Rotary Positional Encoding (RoPE).
    Uses standard PyTorch backend for attention calculation.
    """

    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = LlamaRotaryEmbedding(config = config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(N, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o_proj(out)


# ==========================================================
# 模块 5: FlashAttention-2(GQA) + AbsPE
# ==========================================================
class FlashAttention2_AbsPE_Internal(nn.Module):
    """
    Grouped-Query Attention (GQA) with internal Absolute Positional Encoding (AbsPE).
    Uses flash_attn_func for calculation.
    """

    def __init__(self, config: FlashAttConfig, causal: bool = False):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("This module requires the flash-attn library.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = config.attention_dropout
        self.causal = causal

        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = x + self.pos_embed(self.position_ids[:, :N])

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_key_value_heads, self.head_dim)

        dropout_p = self.dropout if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=self.causal)

        out = out.reshape(B, N, self.hidden_size)
        return self.o_proj(out)
# ==========================================================
# 模块 6: FlashAttention-2(MHSA) + AbsPE
# ==========================================================
class FlashAttention2_MHSA_AbsPE(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) with internal Absolute Positional Encoding (AbsPE)
    using FlashAttention.
    """

    def __init__(self, config: FlashAttConfig, causal: bool = False):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("This module requires the flash-attn library.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = config.attention_dropout
        self.causal = causal

        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        # 加绝对位置编码
        x = x + self.pos_embed(self.position_ids[:, :N])

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        dropout_p = self.dropout if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=self.causal)

        out = out.reshape(B, N, self.hidden_size)
        return self.o_proj(out)


if __name__ == '__main__':
    # 检查 CUDA 是否可用，并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: CUDA not available, running on CPU. FlashAttention will be skipped.")

    # --- 新增: 定义目标数据类型 ---
    # 推荐在支持的GPU上使用 bfloat16
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 配置参数
    config = FlashAttConfig()

    # 模拟输入，并将其创建为正确的设备和数据类型
    dummy_input = torch.randn(4, 100, config.hidden_size, device=device, dtype=dtype)
    print(f"Running on device: {device} with dtype: {dtype}")
    print(f"Input shape: {dummy_input.shape}\n")

    # --- 0. Testing GQA + AbsPE (PyTorch) ---
    print("--- 1. Testing MHSA + AbsPE (PyTorch) ---")
    # 将模型也移动到设备并转换数据类型
    model01 = MHSA_AbsPE_Internal(config).to(device=device, dtype=dtype)
    model01.eval()
    with torch.no_grad():
        output1 = model01(dummy_input)
    print(f"Output shape: {output1.shape}\n")

    # --- 0. Testing GQA + AbsPE (PyTorch) ---
    print("--- 2. Testing MHSA + + RoPE (PyTorch) ---")
    # 将模型也移动到设备并转换数据类型
    model02 = MHSA_RoPE_Internal(config).to(device=device, dtype=dtype)
    model02.eval()
    with torch.no_grad():
        output1 = model02(dummy_input)
    print(f"Output shape: {output1.shape}\n")

    # --- 1. Testing GQA + AbsPE (PyTorch) ---
    print("--- 3. Testing GQA + AbsPE (PyTorch) ---")
    # 将模型也移动到设备并转换数据类型
    model1 = GQA_AbsPE_Internal(config).to(device=device, dtype=dtype)
    model1.eval()
    with torch.no_grad():
        output1 = model1(dummy_input)
    print(f"Output shape: {output1.shape}\n")

    # --- 2. Testing GQA + RoPE (PyTorch) ---
    print("--- 4. Testing GQA + RoPE (PyTorch) ---")
    model2 = GQA_RoPE_Internal(config).to(device=device, dtype=dtype)
    model2.eval()
    with torch.no_grad():
        output2 = model2(dummy_input)
    print(f"Output shape: {output2.shape}\n")

    # --- 3. Testing GQA + AbsPE (FlashAttention) ---
    if device.type == 'cuda' and FLASH_ATTN_AVAILABLE:
        print("--- 5. Testing GQA + AbsPE (FlashAttention) ---")
        model3 = FlashAttention2_AbsPE_Internal(config).to(device=device, dtype=dtype)
        model3.eval()
        with torch.no_grad():
            output3 = model3(dummy_input)
        print(f"Output shape: {output3.shape}\n")

        print("--- 6. Testing MHSA + AbsPE (FlashAttention) ---")
        model4 = FlashAttention2_MHSA_AbsPE(config).to(device=device, dtype=dtype)
        model4.eval()
        with torch.no_grad():
            output4 = model4(dummy_input)
        print(f"Output shape: {output4.shape}\n")




    elif not FLASH_ATTN_AVAILABLE:
        print("--- 3. Skipped GQA + AbsPE (FlashAttention): flash-attn not installed. ---")
    else:
        print("--- 3. Skipped GQA + AbsPE (FlashAttention): No CUDA device found. ---")
