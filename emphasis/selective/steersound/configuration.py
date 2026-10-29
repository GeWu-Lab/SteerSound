from dataclasses import dataclass,asdict

@dataclass
class Edit_FluxParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    patchify_proj_sharing_weight: bool
    sharing_weight: bool
    mean_txt: bool
    enable_flash_attn: bool
    vol_embed: bool
    Pitch_embed: bool

@dataclass
class AutoEncoderParams:
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list
    num_res_blocks: int
    z_channels: int
    enable_flash_attn: bool

ae_params = AutoEncoderParams(in_channels=1, ch=256, out_ch=1, ch_mult=[1, 2, 4, 4], num_res_blocks=4, z_channels=16, enable_flash_attn=False)

dit_params = Edit_FluxParams(in_channels=64, out_channels=64, context_in_dim=768, hidden_size=1536, mlp_ratio=4.0, num_heads=12, depth=2, depth_single_blocks=4, axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=True, vol_embed=True, Pitch_embed=True, patchify_proj_sharing_weight=True, sharing_weight=True, mean_txt=True, enable_flash_attn=False)
