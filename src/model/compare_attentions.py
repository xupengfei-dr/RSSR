# 文件: final_attention_modules.py
import warnings

import torch
import torch.nn as nn
from typing import Optional

# 从辅助文件中导入所有必要的组件
from attention_helpers import FlashAttConfig, LlamaRotaryEmbedding, apply_rotary_pos_emb

# 尝试导入 flash-attn
try:
    from flash_attn import flash_attn_func

    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True

except ImportError:
    print("Warning: flash-attn library not found. FlashAttention modules will not be available.")
    FLASH_ATTN_AVAILABLE = False


# ====================================================================
# 模块 1: 用于表格中的 "MHSA[41] + AbsPE[42]"
# ====================================================================
class Table_MHSA_AbsPE(nn.Module):
    """
    [实验配置 2: MHSA + AbsPE]
    严格的 MHSA 实现 (Q, K, V 头数相同)，内部添加可学习的绝对位置编码。
    使用标准的 PyTorch nn.MultiheadAttention 后端。
    """

    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads  # 强制使用Q的头数以实现MHSA

        self.attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, feature: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = feature.shape

        # 1. 在层内部添加绝对位置编码
        pos_embedding = self.pos_embed(self.position_ids[:, :N])
        x = feature + pos_embedding

        # 2. 准备掩码
        # nn.MultiheadAttention 需要一个布尔掩码，其中 True 表示“需要被忽略”的位置
        # 假设输入的 attention_mask 中 0 表示 padding
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # 3. 执行注意力计算
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return attn_output


# ====================================================================
# 模块 2: 用于表格中的 "MHSA[41] + RoPE[15]"
# ====================================================================
class Table_MHSA_RoPE(nn.Module):
    """
    [实验配置 3: MHSA + RoPE]
    严格的 MHSA 实现，内部应用 LlamaRoPE。
    使用标准的 PyTorch 后端。
    """

    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, feature: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = feature.shape

        q = self.q_proj(feature).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(feature).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(feature).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(N, device=feature.device).unsqueeze(0)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            # 将 (B, N) 的掩码广播到 (B, 1, 1, N)
            mask = attention_mask.view(B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o_proj(out)


# ====================================================================
# 模块 3: 用于表格中的 "FlashAttention2[14] + AbsPE[42]"
# ====================================================================
class Table_FlashAttention2_AbsPE_(nn.Module):
    """
    [实验配置 4: FlashAttention2 + AbsPE]
    严格的 MHSA 实现，内部添加绝对位置编码。
    使用 FlashAttention2 后端进行计算。
    """

    def __init__(self, config: FlashAttConfig, causal: bool = False):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("This module requires the flash-attn library.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = config.attention_dropout
        self.causal = causal

        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, feature: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            warnings.warn("The standard `flash_attn_func` does not support arbitrary padding masks. "
                          "The provided `attention_mask` will be ignored. For handling padding, "
                          "consider using `flash_attn_varlen_func` by unpadding the input first.",
                          UserWarning)

        B, N, _ = feature.shape
        x = feature + self.pos_embed(self.position_ids[:, :N])

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        dropout_p = self.dropout if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=self.causal)

        out = out.reshape(B, N, self.hidden_size)
        return self.o_proj(out)


# ====================================================================
# 模块 3 (已修正): FlashAttention2 + AbsPE with Padding Support
# ====================================================================
class Table_FlashAttention2_AbsPE__(nn.Module):
    """
    [实验配置 4: FlashAttention2 + AbsPE]
    严格的 MHSA 实现，内部添加绝对位置编码。
    使用 FlashAttention2 后端，并正确处理 padding。
    """

    def __init__(self, config: FlashAttConfig, causal: bool = False):
        super().__init__()
        # ... __init__ 方法的其他部分保持不变 ...
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("This module requires the flash-attn library.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = config.attention_dropout
        self.causal = causal

        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, feature: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = feature.shape
        x = feature + self.pos_embed(self.position_ids[:, :N])

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # --- FlashAttention Padding 处理逻辑 ---
        # 检查是否提供了掩码
        if attention_mask is None:
            # 如果没有掩码，所有序列都是满的，可以直接用 flash_attn_func
            dropout_p = self.dropout if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=self.causal)
        else:
            # 如果有掩码，需要使用 varlen 版本
            # 1. 找到每个序列的真实长度
            # attention_mask 中 1 表示有效 token
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)

            # 2. "Unpad" 输入: 将 (B, N, H, D) -> (total_tokens, H, D)
            #    其中 total_tokens 是 batch 中所有有效 token 的总和
            # q_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(q, attention_mask)
            q_unpad, indices, cu_seqlens = unpad_input(q, attention_mask)
            k_unpad, _, _, _ = unpad_input(k, attention_mask)
            v_unpad, _, _, _ = unpad_input(v, attention_mask)

            # 3. 调用 flash_attn_varlen_func
            dropout_p = self.dropout if self.training else 0.0
            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens,
                cu_seqlens,
                N,
                N,
                dropout_p=dropout_p,
                causal=self.causal
            )

            # 4. "Pad" 输出: 将 (total_tokens, H, D) -> (B, N, H, D)
            #    恢复原始的 padding 形状
            out = pad_input(out_unpad, indices, B, N)

        # --- 结束 ---

        out = out.reshape(B, N, self.hidden_size)
        return self.o_proj(out)


# ====================================================================
# 模块 3 (最终修正版): FlashAttention2 + AbsPE with Padding Support
# ====================================================================
class Table_FlashAttention2_AbsPE(nn.Module):
    """
    [实验配置 4: FlashAttention2 + AbsPE]
    严格的 MHSA 实现，内部添加绝对位置编码。
    使用 FlashAttention2 后端，并正确处理 padding。
    """

    def __init__(self, config: FlashAttConfig, causal: bool = False):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("This module requires the flash-attn library.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = config.attention_dropout
        self.causal = causal

        self.pos_embed = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, feature: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = feature.shape
        x = feature + self.pos_embed(self.position_ids[:, :N])

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        if attention_mask is None:
            dropout_p = self.dropout if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=self.causal)
        else:
            # ✅✅✅ --- 最终修正 --- ✅✅✅
            # 1. 改回解包 4 个返回值
            q_unpad, indices, cu_seqlens, max_s = unpad_input(q, attention_mask)
            k_unpad, _, _, _ = unpad_input(k, attention_mask)
            v_unpad, _, _, _ = unpad_input(v, attention_mask)

            dropout_p = self.dropout if self.training else 0.0

            # 2. 使用 unpad_input 返回的 max_s
            out_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens, cu_seqlens,
                max_s, max_s,  # 使用 max_s
                dropout_p=dropout_p,
                causal=self.causal
            )

            out = pad_input(out_unpad, indices, B, N)

        out = out.reshape(B, N, self.hidden_size)
        return self.o_proj(out)

# ====================================================================
# 测试代码
# ====================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Running on device: {device} with dtype: {dtype}\n")

    config = FlashAttConfig()
    dummy_input = torch.randn(4, 100, config.hidden_size, device=device, dtype=dtype)
    # 模拟一个padding mask，最后10个token是padding
    dummy_mask = torch.ones(4, 100, device=device, dtype=dtype)
    dummy_mask[:, -10:] = 0

    print(f"Input shape: {dummy_input.shape}")
    print(f"Mask shape: {dummy_mask.shape}\n")

    models_to_test = {
        "MHSA + AbsPE": Table_MHSA_AbsPE(config),
        "MHSA + RoPE": Table_MHSA_RoPE(config),
    }
    if FLASH_ATTN_AVAILABLE and device.type == 'cuda':
        models_to_test["FlashAttention2 + AbsPE"] = Table_FlashAttention2_AbsPE(config)

    for name, model in models_to_test.items():
        print(f"--- Testing: {name} ---")
        model.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            output = model(dummy_input, attention_mask=dummy_mask)
        print(f"Output shape: {output.shape}\n")