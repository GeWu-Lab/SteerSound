"""Experiment-3 DiT derived from the newest Main-A continuation model.

The audio streams and task/volume/pitch conditioning are retained.  Text is a
frozen T5-base sequence projected by ``txt_in``.  Qwen, Meta-Q, CLAP condition
vectors, connectors, classifiers and discriminators are not part of this file.

"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from ..configuration import Edit_FluxParams

try:
    from flash_attn import flash_attn_interface
except ImportError:
    flash_attn_interface = None


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor,
    scale,
    enable_flash_attn,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    q, k = apply_rope(q, k, pe)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=q.device, dtype=torch.bool)
        if attention_mask.shape != (q.shape[0], q.shape[2]):
            raise ValueError(
                f"attention_mask={tuple(attention_mask.shape)} but q={tuple(q.shape)}"
            )
    if enable_flash_attn and flash_attn_interface is not None:
        # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if attention_mask is not None and not bool(attention_mask.all()):
            batch, length, heads, dim = q.shape
            indices = attention_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            lengths = attention_mask.sum(dim=1, dtype=torch.int32)
            cu_seqlens = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
            q_unpad = q.reshape(batch * length, heads, dim).index_select(0, indices)
            k_unpad = k.reshape(batch * length, heads, dim).index_select(0, indices)
            v_unpad = v.reshape(batch * length, heads, dim).index_select(0, indices)
            x_unpad = flash_attn_interface.flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens,
                cu_seqlens,
                int(lengths.max().item()),
                int(lengths.max().item()),
                dropout_p=0.0,
                softmax_scale=scale,
                causal=False,
            )
            x_flat = torch.zeros_like(q.reshape(batch * length, heads, dim))
            x_flat.index_copy_(0, indices, x_unpad)
            x = x_flat.reshape(batch, length, heads, dim)
        else:
            x = flash_attn_interface.flash_attn_func(
                q, k, v,
                softmax_scale=scale,
                causal=False,
            )
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "B L H D -> B L (H D)")
        return x
    else:
        sdpa_mask = attention_mask[:, None, None, :] if attention_mask is not None else None
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        if attention_mask is not None:
            x = x * attention_mask[:, None, :, None]
        x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, ratio):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, int(ratio * hidden_dim), bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(int(ratio * hidden_dim), hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, enable_flash_attn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
        self.enable_flash_attn = enable_flash_attn

    def forward(self, x: Tensor, pe: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(
            q, k, v, pe=pe, enable_flash_attn=self.enable_flash_attn,
            scale=None, attention_mask=attention_mask,
        )
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, enable_flash_attn=False):
        super().__init__()
        self.enable_flash_attn = enable_flash_attn
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, enable_flash_attn=enable_flash_attn)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, enable_flash_attn=enable_flash_attn)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        txt_mask: Optional[Tensor] = None,
    ):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        combined_mask = None
        if txt_mask is not None:
            combined_mask = torch.cat(
                [txt_mask.to(torch.bool), torch.ones(img.shape[:2], device=img.device, dtype=torch.bool)],
                dim=1,
            )
        attn = attention(
            q, k, v, pe=pe, scale=self.scale,
            enable_flash_attn=self.enable_flash_attn,
            attention_mask=combined_mask,
        )
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, enable_flash_attn=False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        self.enable_flash_attn = enable_flash_attn

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(
            q, k, v, pe=pe, scale=self.scale,
            enable_flash_attn=self.enable_flash_attn,
            attention_mask=attention_mask,
        )
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Edit_Flux(nn.Module):
    """整体训练模型 DiT.

    Experiment 3 keeps the current continuation model's audio, task, volume
    and pitch pathways, but replaces Qwen/Meta-Q with direct frozen T5 text.
    CLAP vectors are intentionally absent.

    参数注入 vec 路径:
        vec  = time_in(timestep_embedding(t, 256))
        vec += task_in(timestep_embedding(task, 256))     # task 走 task/5 浮点
        vec += vloume_in(timestep_embedding(change_vol, 256))
        vec += Pitch_in(timestep_embedding(change_Pitch, 256))
    """
    def __init__(self, params: Edit_FluxParams):
        super().__init__()
        self.params = params
        self.patchify_proj_sharing_weight = params.patchify_proj_sharing_weight
        self.sharing_weight = params.sharing_weight
        self.mean_txt = params.mean_txt
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        if self.patchify_proj_sharing_weight:
            self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        else:
            self.ref_img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
            self.noi_img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, ratio=params.mlp_ratio)
        # Renamed from {guidance_in, params.guidance_embed}
        self.task_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, ratio=params.mlp_ratio)
            if params.guidance_embed else nn.Identity()
        )
        self.vloume_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, ratio=params.mlp_ratio)
            if params.vol_embed else nn.Identity()
        )
        self.Pitch_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, ratio=params.mlp_ratio)
            if params.Pitch_embed else nn.Identity()
        )

        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)
        self.enable_flash_attn = params.enable_flash_attn

        if self.sharing_weight:
            self.pre_double_blocks = nn.ModuleList([
                DoubleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth)
            ])
            self.double_blocks = nn.ModuleList([
                DoubleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth)
            ])
            self.pre_single_blocks = nn.ModuleList([
                SingleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth_single_blocks)
            ])
            self.single_blocks = nn.ModuleList([
                SingleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth_single_blocks)
            ])
        else:
            self.noi_ref_double_blocks = nn.ModuleList([
                DoubleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth)
            ])
            self.txt_noi_double_blocks = nn.ModuleList([
                DoubleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth)
            ])
            self.txt_ref_double_blocks = nn.ModuleList([
                DoubleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth)
            ])
            self.noi_ref_single_blocks = nn.ModuleList([
                SingleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth_single_blocks)
            ])
            self.txt_noi_single_blocks = nn.ModuleList([
                SingleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth_single_blocks)
            ])
            self.txt_ref_single_blocks = nn.ModuleList([
                SingleStreamBlock(self.hidden_size, self.num_heads,
                                  mlp_ratio=params.mlp_ratio,
                                  enable_flash_attn=params.enable_flash_attn)
                for _ in range(params.depth_single_blocks)
            ])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        ref_img: Tensor,
        noi_img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        txt_mask: Optional[Tensor] = None,
        task: Optional[Tensor] = None,
        change_vol=None,
        change_Pitch=None,
        **kwargs,
    ) -> Tensor:
        if ref_img.ndim != 3 or txt.ndim != 3 or noi_img.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        if self.patchify_proj_sharing_weight:
            noi_img = self.img_in(noi_img)
            ref_img = self.img_in(ref_img)
        else:
            noi_img = self.noi_img_in(noi_img)
            ref_img = self.ref_img_in(ref_img)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:  # 即 task_embed
            if task is None:
                raise ValueError("Didn't get task.")
            vec = vec + self.task_in(timestep_embedding(task, 256))
        if self.params.vol_embed:
            if change_vol is None:
                raise ValueError("Didn't get change_vol.")
            vec = vec + self.vloume_in(timestep_embedding(change_vol, 256))
        if self.params.Pitch_embed:
            if change_Pitch is None:
                raise ValueError("Didn't get change_Pitch.")
            vec = vec + self.Pitch_in(timestep_embedding(change_Pitch, 256))

        txt = self.txt_in(txt)
        if txt_mask is None:
            txt_mask = torch.ones(txt.shape[:2], device=txt.device, dtype=torch.bool)
        else:
            txt_mask = txt_mask.to(device=txt.device, dtype=torch.bool)
            if txt_mask.shape != txt.shape[:2]:
                raise ValueError(
                    f"txt_mask={tuple(txt_mask.shape)} but txt={tuple(txt.shape)}"
                )

        if self.sharing_weight:
            txt_total = torch.cat([txt, txt], dim=0)
            txt_mask_total = torch.cat([txt_mask, txt_mask], dim=0)
        else:
            noi_txt = txt.clone()
            ref_txt = txt
        txt_img_ids = torch.cat((txt_ids, img_ids), dim=1)
        img_img_ids = torch.cat((img_ids, img_ids), dim=1)

        txt_img_pe = self.pe_embedder(txt_img_ids)
        img_img_pe = self.pe_embedder(img_img_ids)

        for block_ids in range(self.params.depth):
            if self.sharing_weight:
                noi_img, ref_img = self.pre_double_blocks[block_ids](
                    img=noi_img, txt=ref_img, vec=vec, pe=img_img_pe
                )
                img_total = torch.cat([noi_img, ref_img], dim=0)
                img_total, txt_total = self.double_blocks[block_ids](
                    img=img_total, txt=txt_total,
                    vec=torch.cat([vec, vec], dim=0),
                    pe=torch.cat([txt_img_pe, txt_img_pe], dim=0),
                    txt_mask=txt_mask_total,
                )
                if self.mean_txt:
                    noi_txt, ref_txt = txt_total.chunk(chunks=2, dim=0)
                    txt = (noi_txt + ref_txt) / 2
                    txt_total = torch.cat([txt, txt], dim=0)
                noi_img, ref_img = img_total.chunk(chunks=2, dim=0)
            else:
                noi_img, ref_img = self.noi_ref_double_blocks[block_ids](
                    img=noi_img, txt=ref_img, vec=vec, pe=img_img_pe
                )
                noi_img, noi_txt = self.txt_noi_double_blocks[block_ids](
                    img=noi_img, txt=noi_txt, vec=vec, pe=txt_img_pe,
                    txt_mask=txt_mask,
                )
                ref_img, ref_txt = self.txt_ref_double_blocks[block_ids](
                    img=ref_img, txt=ref_txt, vec=vec, pe=txt_img_pe,
                    txt_mask=txt_mask,
                )
                if self.mean_txt:
                    txt = (noi_txt + ref_txt) / 2
                    noi_txt = txt.clone()
                    ref_txt = txt

        for block_ids in range(self.params.depth_single_blocks):
            if self.sharing_weight:
                img = torch.cat([noi_img, ref_img], 1)
                img = self.pre_single_blocks[block_ids](img, vec=vec, pe=img_img_pe)
                noi_img, ref_img = img.chunk(chunks=2, dim=1)
                img_total = torch.cat([noi_img, ref_img], dim=0)
                txt_img = torch.cat([txt_total, img_total], dim=1)
                txt_img = self.single_blocks[block_ids](
                    txt_img,
                    vec=torch.cat([vec, vec], dim=0),
                    pe=torch.cat([txt_img_pe, txt_img_pe], dim=0),
                    attention_mask=torch.cat(
                        [
                            txt_mask_total,
                            torch.ones(img_total.shape[:2], device=img_total.device, dtype=torch.bool),
                        ],
                        dim=1,
                    ),
                )
                txt_total = txt_img[:, :txt_total.shape[1], :]
                img_total = txt_img[:, txt_total.shape[1]:, :]
                noi_img, ref_img = img_total.chunk(chunks=2, dim=0)
                if self.mean_txt:
                    noi_txt, ref_txt = txt_total.chunk(chunks=2, dim=0)
                    txt = (noi_txt + ref_txt) / 2
                    txt_total = torch.cat([txt, txt], dim=0)
            else:
                img = torch.cat([noi_img, ref_img], 1)
                img = self.noi_ref_single_blocks[block_ids](img, vec=vec, pe=img_img_pe)
                noi_img, ref_img = img.chunk(chunks=2, dim=1)
                noi_txt_noi_img = torch.cat([noi_txt, noi_img], 1)
                noi_txt_noi_img = self.txt_noi_single_blocks[block_ids](
                    noi_txt_noi_img, vec=vec, pe=txt_img_pe,
                    attention_mask=torch.cat(
                        [
                            txt_mask,
                            torch.ones(noi_img.shape[:2], device=noi_img.device, dtype=torch.bool),
                        ],
                        dim=1,
                    ),
                )
                noi_txt = noi_txt_noi_img[:, :noi_txt.shape[1], :]
                noi_img = noi_txt_noi_img[:, noi_txt.shape[1]:, :]
                ref_txt_ref_img = torch.cat([ref_txt, ref_img], 1)
                ref_txt_ref_img = self.txt_ref_single_blocks[block_ids](
                    ref_txt_ref_img, vec=vec, pe=txt_img_pe,
                    attention_mask=torch.cat(
                        [
                            txt_mask,
                            torch.ones(ref_img.shape[:2], device=ref_img.device, dtype=torch.bool),
                        ],
                        dim=1,
                    ),
                )
                ref_txt = ref_txt_ref_img[:, :ref_txt.shape[1], :]
                ref_img = ref_txt_ref_img[:, ref_txt.shape[1]:, :]
                if self.mean_txt:
                    txt = (noi_txt + ref_txt) / 2
                    noi_txt = txt.clone()
                    ref_txt = txt

        noi_img = self.final_layer(noi_img, vec)
        return noi_img
