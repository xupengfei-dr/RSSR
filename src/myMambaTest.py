"""

adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
It implements a Mamba2 model BUT still relies on the official Triton code for Mamba2.
(Coming soon hopefully is a full torch version, like in mamba.py and pscan.py)

Also, the file implements a caching mecanism, and a config similar to what's being done in mamba.py, as well as supports muP.

When passing an input of length 1 in the forward function, the model automatically routes the call to the step function.
This step function does one "classic" RNN-like step of computation, using the input and the cache provided. It returns the output as well as the new cache.
This is used for inference, to generate tokens one by one.

Also, this model supports prefilling a prompt and decoding token by token : this is just a mix of forwarding (in parallel) the input, and then decoding step by step.
In order to do that, we need the forward/parallel call (also used in training) to output the cache, then used to start the step by step decoding part.
To use this mode, you need to call Mamba2.forward() with your input (of shape (B, L, D)) as well as with a non-None cache (like full zeros, it doesnt matter, just not "None").
The forward call will thus return the output and the cache. From there, you can start your step by step decoding (see just above).

The cache is composed of two objects :
-h_cache: the last hidden state. Just like an RNN : you have to keep track of only the last h.
-conv_state: because Mamba2 uses a convolution over the time sequence, with a filter of length d_conv=4, you have to keep the last d_conv-1=3 inputs of that convolution to be able to run it provided a new input.

h_cache is of shape (B, n_heads, d_head, N) and is initialized at 0 (ie no starting hidden state, which is the default behavior in Mamba).
conv_state is of shape (B, EDN * 2*n_groups*, d_conv) and is initialized at 0

(B=batch_size, L=seq len, E = expand_factor, D=d_model, N=d_state)



(TODO: make it full pytorch (ie translate mamba_chunk_scan_combined in pytorch))



 torch.cuda.set_device(6)
    devie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = Mamba2Config(d_model=768,n_layers=1,d_head=64,)
    model = Mamba2(model_config).to(devie)
    # 随机生成一个形状为 (64, 40, 768) 的张量
    tensor = torch.rand(64, 40, 768).to(devie)

    result = model(tensor)

    print(result.shape)


"""

import math
from dataclasses import dataclass
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from transformers.activations import ACT2FN, logger
from transformers.models.mamba2.modeling_mamba2 import Mamba2Output

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

except ImportError:
    RMSNormGated, LayerNorm = None, None

    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None
    selective_state_update = None

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


@dataclass
class Mamba2Config:
    #  todo : plutot n_heads non ?
    d_state: int = 64  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4
    n_groups: int = 1  #  todo : ??

    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple = (0.0, float("inf"))
    conv_init = None

    learnable_init_states: bool = False
    activation: str = "swish"  #  "swish" or "silu"

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = True
    conv_bias: bool = True

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    chunk_size: int = 256
    use_mem_eff_path: bool = True
    dtype = None
    device = None

    # def __init__(self):
    #     self.num_hidden_layers = 12
    #     self.use_bias = True
    #     self.time_step_max = None
    #     self.time_step_min = None
    #     self.time_step_limit = None
    #     self.head_dim = None
    #     self.rms_norm = True
    #     self.fused_add_norm =True
    #     self.pad_vocab_size_multiple =16
    #     self.tie_embeddings =True
    #     self.layer_norm_epsilon = None
    #     self.hidden_act = None
    #     self.use_conv_bias = None
    #     self.time_step_rank = None
    #     self.expand = 4
    #     self.state_size = None
    #     self.hidden_size = 768
    #     self.num_heads = 12
    #     self.conv_kernel = None
    #     self.n_layers=12
    #     self.d_model =768
    #     self.d_head=64
    #     self.residual_in_fp32 = True
    #     self.d_intermediate = 0
    #     self.vocab_size =50277

    def __init__(self):
        self.num_hidden_layers = 12  # 根据你的任务，保持12层
        self.use_bias = True
        self.time_step_max = 0.1
        self.time_step_min = 0.0001
        self.time_step_rank = 256
        self.time_step_limit = [0.0, float('inf')]
        self.time_step_floor=0.0001

        self.head_dim = 64  # 每个头的维度，通常是 64 或 128
        self.rms_norm = True  # 使用 RMSNorm
        self.fused_add_norm = True  # 融合加法和归一化
        self.pad_vocab_size_multiple = 16  # 填充词汇大小为16的倍数
        self.tie_embeddings = True  # 词嵌入共享权重
        self.layer_norm_epsilon = 1e-5  # 通常选择一个小的 epsilon 值
        self.hidden_act = "silu"  # 激活函数，gelu 是常用的选择
        self.use_conv_bias = True
        self.expand = 2  # 每个层的扩展比例
        self.state_size = 128
        self.hidden_size = 768  # 隐藏层的维度
        self.num_heads = 24  # 每个层的注意力头数
        self.conv_kernel = 4
        self.n_layers = 12  # 模型层数
        self.d_model = 768  # 模型的维度
        self.d_head = 64  # 每个头的维度
        self.residual_in_fp32 = True
        self.d_intermediate = 0  # 如果是用于前馈网络，可以设置更大的值
        self.intermediate_size=1536
        self.vocab_size = 30522  # 词汇表大小
        self.layer_norm_epsilon = 1e-5  # 防止归一化时出现数值问题

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0

        assert (self.d_inner / self.d_head) % 8 == 0, "requierement of causal_conv1d"

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba2Cache:
    """
    Arguments:
        config: Mamba2Config
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
            self, config: Mamba2Config, batch_size: int, dtype: torch.dtype = torch.float16,
            device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.n_groups * config.state_size,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(
                batch_size, config.num_heads, config.head_dim, config.state_size, device=device, dtype=dtype
            )
            for i in range(config.num_hidden_layers)
        }
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

    def update_conv_state(
            self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class Mamba2Model(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        # self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Mamba2Output]:
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            # gradient_checkpointing =False
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.use_bias = False

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
            self,
            hidden_states: torch.Tensor,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # getting projected states from cache if it exists
        if cache_params is not None and cache_params.seqlen_offset > 0:
            in_projected_states = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
            _, _, gate, hidden_states_B_C, dt = torch.split(in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())  # (nheads,)

            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        # if no cache is found, calling the kernel
        else:
            if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            # 1. Gated MLP's linear projection
            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            if self.training and cache_params is None:
                out, ssm_state = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                # 1D Convolution
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
                    )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)[:, :seq_len]
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    time_step,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[Mamba2Cache] = None,
                      cache_position: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # Gated MLP's linear projection
        projected_states = self.in_proj(input_states.squeeze(1))
        d_mlp = (projected_states.shape[
                     -1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]  # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                # handle batched generation - states are copied through
                conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype)[:, None,
                                ...]  # [batch, 1, intermediate_size] : decoding
            else:
                hidden_states = hidden_states.transpose(1, 2)
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states).transpose(1, 2))[:, :seq_len,
                                :]  # [batch, intermediate_size, seq_len]
                if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                    dtype = hidden_states.dtype
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size,
                                                          self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())  # [num_heads]
        if cache_params is not None and cache_params.seqlen_offset > 0:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_min)  # , self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = torch.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim,
                                                  self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in
                                      (hidden_states, A, B, C)]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None] * hidden_states.permute(0, 1, 3, 2, 4)[...,
                                                                              None, :]).sum(dim=3).permute(0, 1, 2, 4,
                                                                                                           3)
            if cache_params is not None and cache_params.seqlen_offset > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            # compute Yoff
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states

    # fmt: on

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)


class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        return hidden_states


class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Mamba2RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


if __name__ == "__main__":
    torch.cuda.set_device(6)
    devie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = Mamba2Config()
    model = Mamba2Model(model_config).to(devie)
    # 随机生成一个形状为 (64, 40, 768) 的张量
    tensor = torch.rand(64, 40, 768).to(devie)

    # result = model(inputs_embeds = tensor)[0]
    result = model(inputs_embeds=tensor)[0]

    print(result.shape)
