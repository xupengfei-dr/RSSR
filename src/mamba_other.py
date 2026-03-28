"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


# 使用dataclass装饰器自动生成初始化方法和类的字符串表示方法
@dataclass
class ModelArgs:
    # @dataclass 会自动为这个类生成初始化方法和代表类的字符串形式的方法
    d_model: int  # 定义模型的隐藏层维度
    n_layer: int  # 定义模型的层数
    vocab_size: int  # 定义词汇表的大小
    d_state: int = 16  # 定义状态空间的维度，默认为16
    expand: int = 2  # 定义扩展因子，默认为2
    dt_rank: Union[int, str] = 'auto'  # 定义输入依赖步长Δ的秩，'auto'表示自动设置
    d_conv: int = 4  # 定义卷积核的维度，默认为4
    pad_vocab_size_multiple: int = 8  # 定义词汇表大小的最小公倍数，默认为8
    conv_bias: bool = True  # 定义卷积层是否使用偏置项
    bias: bool = False  # 定义其他层（如线性层）是否使用偏置项

    def __post_init__(self):
        # 在__init__后自动被调用，用于执行初始化之后的额外设置或验证
        # 计算内部维度，即扩展后的维度
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':  # 如果dt_rank未指定，则自动计算设置
            # 根据隐藏层维度自动计算Δ的秩
            self.dt_rank = math.ceil(self.d_model / 16)
        # 确保vocab_size是pad_vocab_size_multiple的倍数
        # 如果不是，调整为最近的倍数
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        # 保存传入的ModelArgs对象，包含模型的配置参数
        self.args = args
        # 创建一个嵌入层，将词汇表中的词转换为对应的向量表示
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        # 创建一个包含多个残差块的模块列表，残差块的数量等于模型层数
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        # 创建一个RMSNorm模块，用于归一化操作
        self.norm_f = RMSNorm(args.d_model)
        # 创建一个线性层，用于最终的输出，将隐藏层的输出映射回词汇表的大小
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        # 将线性层的输出权重与嵌入层的权重绑定，这是权重共享的一种形式，有助于减少参数数量并可能提高模型的泛化能力
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
        # See "Weight Tying" paper

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        # 将输入ID转换为向量表示
        x = self.embedding(input_ids)
        # 遍历所有的残差块，并应用它们
        for layer in self.layers:
            x = layer(x)
        # 应用归一化操作
        x = self.norm_f(x)
        # 通过线性层得到最终的logits输出
        logits = self.lm_head(x)
        # 返回模型的输出
        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        # 保存传入的ModelArgs对象，包含模型的配置参数
        self.args = args
        # 创建一个MambaBlock，它是这个残差块的核心组件
        self.mixer = MambaBlock(args)
        # 创建一个RMSNorm归一化模块，用于归一化操作
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
             x (Tensor): 输入张量，形状为(batch_size, sequence_length, hidden_size)
        Returns:
            output: shape (b, l, d)
            输出张量，形状与输入相同
        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        # 应用归一化和MambaBlock，然后与输入x进行残差连接
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        # 保存模型参数
        self.args = args
        # 输入线性变换层
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # 创建了一个所谓的“深度卷积”，其中每个输入通道被单独卷积到每个输出通道。
        # 这意味着每个输出通道的结果是通过仅与一个输入通道卷积得到的。
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        # 将输入x映射到状态空间模型的参数Δ、B和C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        # 将Δ从args.dt_rank维度映射到args.d_inner维度
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        # 创建一个重复的序列，用于初始化状态空间模型的矩阵A
        # n->dxn
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        # 将矩阵A的对数值作为可训练参数保存
        self.A_log = nn.Parameter(torch.log(A))
        # 初始化矩阵D为全1的可训练参数
        self.D = nn.Parameter(torch.ones(args.d_inner))
        # 输出线性变换层
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """MambaBlock的前向传播函数，与Mamba论文图3 Section 3.4相同.

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        # 获取输入x的维度
        # batchsize,seq_len,dim
        (b, l, d) = x.shape  # 获取输入x的维度
        # 应用输入线性变换
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        # 将变换后的输出分为两部分x和res。
        # 得到的x分为两个部分，一部分x继续用于后续变换，生成所需要的参数，res用于残差部分
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        # 调整x的形状
        x = rearrange(x, 'b l d_in -> b d_in l')
        # 应用深度卷积，然后截取前l个输出
        x = self.conv1d(x)[:, :, :l]
        # 再次调整x的形状
        x = rearrange(x, 'b d_in l -> b l d_in')
        # 应用SiLU激活函数
        x = F.silu(x)
        # 运行状态空间模型
        y = self.ssm(x)
        # 将res的SiLU激活结果与y相乘
        y = y * F.silu(res)
        # 应用输出线性变换
        output = self.out_proj(y)
        # 返回输出结果
        return output

    def ssm(self, x):
        """运行状态空间模型，参考Mamba论文 Section 3.2和注释[2]:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        # 获取A_log的维度
        # A在初始化时候经过如下赋值：
        #  A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        #  self.A_log = nn.Parameter(torch.log(A))
        # （args.d_inner, args.d_state）
        (d_in, n) = self.A_log.shape  # 获取A_log的维度

        # 计算 ∆ A B C D, 这些属于状态空间参数.
        #     A, D 是 与输入无关的 (见Mamba论文Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C 与输入有关(这是与线性是不变模型S4最大的不同,
        #                       也是为什么Mamba被称为 “选择性” 状态空间的原因)

        # 计算矩阵A
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        # 取D的值
        D = self.D.float()

        # 应用x的投影变换
        # ( b,l,d_in) -> (b, l, dt_rank + 2*n)
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        # 分割delta, B, C
        # delta: (b, l, dt_rank). B, C: (b, l, n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        # 应用dt_proj并计算delta
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        # 应用选择性扫描算法
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """执行选择性扫描算法，参考Mamba论文[1] Section 2和注释[2]. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        经典的离散状态空间公式:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
       除了B和C (以及step size delta用于离散化) 与输入x(t)相关.

        参数:
            u: shape (b, l, d_in)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        过程概述：

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        # 获取输入u的维度
        (b, l, d_in) = u.shape
        # 获取矩阵A的列数
        n = A.shape[1]  # A: shape (d_in, n)

        # 离散化连续参数(A, B)
        # - A 使用 zero-order hold (ZOH) 离散化 (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is 使用一个简化的Euler discretization而不是ZOH.根据作者的讨论:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"

        # 计算离散化的A
        # 将delta和A进行点乘，将A沿着delta的最后一个维度进行广播，然后执行逐元素乘法
        # A:(d_in, n),delta:(b, l, d_in)
        # A广播拓展->(b,l,d_in, n)，deltaA对应原论文中的A_bar
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        # delta、B和u,这个计算和原始论文不同
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        # 执行选择性扫描,初始化状态x为零
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        # 初始化输出列表ys
        ys = []
        for i in range(l):
            # 更新状态x
            # deltaA:((b,l,d_in, n)
            # deltaB_u:( b,l,d_in,n)
            # x:
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # 计算输出y
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            # 将输出y添加到列表ys中
            ys.append(y)
        # 将列表ys堆叠成张量y
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        # 将输入u乘以D并加到输出y上
        y = y + u * D

        return y


class RMSNorm(nn.Module):
    """
    初始化RMSNorm模块，该模块实现了基于均方根的归一化操作。

    参数:
    d_model (int): 模型的特征维度。
    eps (float, 可选): 为了避免除以零，添加到分母中的一个小的常数。
    """

    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 保存输入的eps值，用于数值稳定性。
        # 创建一个可训练的权重参数，初始值为全1，维度与输入特征维度d_model相同。
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
                计算输入x的均方根值，用于后续的归一化操作。
                x.pow(2) 计算x中每个元素的平方。
                mean(-1, keepdim=True) 对x的最后一个维度（特征维度）进行平方和求平均，保持维度以便进行广播操作。
                torch.rsqrt 对求得的平均值取倒数和平方根，得到每个特征的均方根值的逆。
                + self.eps 添加一个小的常数eps以保持数值稳定性，防止除以零的情况发生。
                x * ... * self.weight 将输入x与计算得到的归一化因子和可训练的权重相乘，得到最终的归一化输出。
                """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
