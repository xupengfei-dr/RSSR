import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import MixerModel
from src.mamba2_other import Mamba2,Mamba2Config
from mamba_ssm.modules.block import Block
# 假设的 Block 模块（从你提供的代码）


# 假设的 Mixer 和 MLP 类
def init_mix():
    # 定义模型的配置
    config = {
        "d_model": 512,  # 模型的嵌入维度（隐藏层维度）
        "n_layer": 12,  # 模型中的层数（例如12层）
        "d_intermediate": 2048,  # 前馈网络中间层的维度
        "vocab_size": 30000,  # 词汇表大小
        "ssm_cfg": None,  # 选择是否使用 SSM（如果为None，则不使用）
        "attn_layer_idx": None,  # 如果需要指定不同的层使用不同的注意力配置
        "attn_cfg": None,  # 注意力配置（如果需要）
        "norm_epsilon": 1e-5,  # 归一化时的 epsilon 参数
        "rms_norm": False,  # 是否使用 RMSNorm
        "initializer_cfg": None,  # 权重初始化配置
        "fused_add_norm": False,  # 是否融合 Add 和 LayerNorm
        "residual_in_fp32": False,  # 残差连接是否使用 FP32
        "device": "cuda",  # 设备设置（CPU 或 GPU）
        "dtype": torch.float32,  # 数据类型（float32）
    }

    # 创建 MixerModel 实例
    mixer_model = MixerModel(
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        d_intermediate=config["d_intermediate"],
        vocab_size=config["vocab_size"],
        ssm_cfg=config["ssm_cfg"],
        attn_layer_idx=config["attn_layer_idx"],
        attn_cfg=config["attn_cfg"],
        norm_epsilon=config["norm_epsilon"],
        rms_norm=config["rms_norm"],
        initializer_cfg=config["initializer_cfg"],
        fused_add_norm=config["fused_add_norm"],
        residual_in_fp32=config["residual_in_fp32"],
        device=config["device"],
        dtype=config["dtype"],
    )

    # 打印模型结构
    print(mixer_model)
    return  mixer_model


class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





if __name__ == "__main__":
    torch.cuda.set_device(6)
    devie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = Mamba2Config(d_model=768,n_layers=1,d_head=64,)
    model = Mamba2(model_config).to(devie)
    # 随机生成一个形状为 (64, 40, 768) 的张量
    tensor = torch.rand(64, 40, 768).to(devie)

    result = model(tensor)

    print(result.shape)

















''' batch_size = 8
    seq_len = 14
    dim = 512
    vocab_size = 30000  # 假设有 10000 个词
    d_model = 512  # 嵌入维度
    devie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # (batch_size, seq_len)
    # 创建嵌入层
    embedding = nn.Embedding(vocab_size, d_model)
    model = Mamba2(d_model=512).to(devie)
    # 通过嵌入层查找输入索引的嵌入表示
    hidden_states = embedding(input_ids).to(devie)  # (batch_size, seq_len, d_model)
    output = model(u=hidden_states).to(devie)
    print(f"Output shape: {output.shape}")  # 应该输出 [batch_size, num_classes]，即 [8, 10]'''









