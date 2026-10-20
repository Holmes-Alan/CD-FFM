"""
flower_layers.py
=================
Verbatim copy (2D-only subset) of the real Flower implementation from
the Flower reference implementation (ref. in the paper),
physically copied here (not imported from that path) per instruction --
this project should not depend on Extreme_weather's directory.

Trimmed from the original: dropped 3D support (custom_grid_sample3d) and
the RK2/Conservative/Spherical SelfWarp variants (not needed for this
comparison round) -- everything KEPT is copied unmodified, not
reimplemented, specifically to avoid reintroducing the bugs/subtleties
(multi-head warping, periodic-boundary grid_sample, pre-norm residual
placement, U-Net divisibility constraints) that an improvised
reimplementation got wrong the first time (see model.py history).

FiLM/InstanceNorm are included even though our usage always passes
use_meta_conditioning=False (so they're never actually instantiated) --
kept so FlowerBlock's body is copied byte-for-byte unmodified, rather than
edited to remove a branch we don't use.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from einops import rearrange


# =============================================================================
# Utility Functions (verbatim)
# =============================================================================

def create_grid(spatial_resolution):
    """Create a coordinate grid in [-1, 1] range."""
    meshgrid = torch.meshgrid(*[torch.linspace(-1., 1., d_i) for d_i in spatial_resolution], indexing='ij')
    grid = torch.stack(meshgrid[::-1], dim=-1)  # reverse the meshgrid list
    return grid


def derive_padding_strategies(bc_list, n_spatial_dims):
    """Convert boundary condition list to padding strategies."""
    if not bc_list:
        return ['zeros' for _ in range(n_spatial_dims)]

    padding = ['periodic' if bc.upper() == 'PERIODIC' else 'zeros' for bc in bc_list]

    if len(padding) == 1:
        padding = [padding[0] for _ in range(n_spatial_dims)]

    return padding


# =============================================================================
# Stochastic depth (DropPath) (verbatim)
# =============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# =============================================================================
# Normalization Layers (verbatim) -- unused in our config (use_meta_conditioning
# is always False here) but kept so FlowerBlock's body needs zero edits.
# =============================================================================

class InstanceNorm(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class FiLM(nn.Module):
    def __init__(self, num_channels, meta_dim=1, norm_type='layer', num_groups=32, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.num_channels = num_channels
        self.meta_dim = meta_dim
        self.norm_type = norm_type.lower()
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (num_channels,)

        if self.norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=False)
        elif self.norm_type == 'instance':
            self.norm = InstanceNorm()
        elif self.norm_type == 'identity':
            self.norm = nn.Identity()
        else:
            raise ValueError(f"norm_type must be 'group', 'layer', 'instance', or 'identity', got {norm_type}")

        self.weight = nn.Linear(meta_dim, num_channels)
        self.bias = nn.Linear(meta_dim, num_channels)

    def forward(self, x, meta=None):
        if self.norm_type in ['group', 'instance']:
            if self.data_format == "channels_last":
                x_for_norm = x.permute(0, -1, *range(1, x.dim() - 1))
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, *range(2, x.dim()), 1)
            else:
                x = self.norm(x)
        elif self.norm_type == 'layer':
            if self.data_format == "channels_last":
                x = self.norm(x)
            else:
                x_for_norm = x.permute(0, *range(2, x.dim()), 1)
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, -1, *range(1, x.dim() - 1))
        elif self.norm_type == 'identity':
            x = self.norm(x)
        else:
            raise ValueError()

        if meta is None:
            return x

        if meta.dim() == 1:
            meta = meta.unsqueeze(-1)

        meta = meta.type_as(x)
        weight = self.weight(meta)
        bias = self.bias(meta)

        if self.data_format == "channels_last":
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(1)
                bias = bias.unsqueeze(1)
            return weight * x + bias
        else:
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            return weight * x + bias


# =============================================================================
# Custom Grid Sampling with Periodic Boundary Support (verbatim, 2D only)
# =============================================================================

def custom_grid_sample2d(
    input, grid, mode="bilinear", padding_modes=("zeros", "zeros"), align_corners=False
):
    """
    Args:
        input: (N, C, H, W)
        grid:  (N, H, W, 2), normalized coords in [-1, 1]
        padding_modes: (padding_h, padding_w), each in {'zeros','border','reflection','periodic'}.
    """
    padding_h, padding_w = padding_modes

    if padding_w.lower() != "periodic" and padding_h.lower() != "periodic":
        if padding_w != padding_h:
            msg = "When using different paddings for x and y dimensions, one should be `periodic`."
            raise ValueError(msg)
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_w, align_corners=align_corners
        )

    periodic_w = padding_w == "periodic"
    periodic_h = padding_h == "periodic"
    final_padding_w = "border" if periodic_w else padding_w
    final_padding_h = "border" if periodic_h else padding_h

    N, C, H, W = input.shape

    if periodic_h and periodic_w:
        top_row = input[:, :, 0:1, :]
        left_col = input[:, :, :, 0:1]
        top_left_corner = input[:, :, 0:1, 0:1]
        padded_h = torch.cat([input, top_row], dim=2)
        left_col_extended = torch.cat([left_col, top_left_corner], dim=2)
        padded_input = torch.cat([padded_h, left_col_extended], dim=3)
        H_padded = H + 1
        W_padded = W + 1
    elif periodic_h:
        top_row = input[:, :, 0:1, :]
        padded_input = torch.cat([input, top_row], dim=2)
        H_padded = H + 1
        W_padded = W
    elif periodic_w:
        left_col = input[:, :, :, 0:1]
        padded_input = torch.cat([input, left_col], dim=3)
        H_padded = H
        W_padded = W + 1
    else:
        padded_input = input
        H_padded = H
        W_padded = W

    gx, gy = grid[..., 0], grid[..., 1]

    if align_corners:
        px = (gx + 1) * (W - 1) / 2
        py = (gy + 1) * (H - 1) / 2

        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py

        gx_new = 2 * px_wrapped / W - 1 if periodic_w else (
            2 * px_wrapped / (W_padded - 1) - 1 if W_padded > 1 else gx)
        gy_new = 2 * py_wrapped / H - 1 if periodic_h else (
            2 * py_wrapped / (H_padded - 1) - 1 if H_padded > 1 else gy)
    else:
        px = ((gx + 1) * W - 1) / 2
        py = ((gy + 1) * H - 1) / 2

        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py

        gx_new = 2 * (px_wrapped + 0.5) / (W + 1) - 1 if periodic_w else (
            2 * (px_wrapped + 0.5) / W_padded - 1 if W_padded > 0 else gx)
        gy_new = 2 * (py_wrapped + 0.5) / (H + 1) - 1 if periodic_h else (
            2 * (py_wrapped + 0.5) / H_padded - 1 if H_padded > 0 else gy)

    grid_new = torch.stack([gx_new, gy_new], dim=-1)

    if periodic_w and periodic_h:
        final_padding = "border"
    elif periodic_w:
        final_padding = final_padding_h
    elif periodic_h:
        final_padding = final_padding_w
    else:
        final_padding = final_padding_w

    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )


# =============================================================================
# Module Dictionaries (verbatim, 2D entries only)
# =============================================================================

conv_modules = {2: nn.Conv2d}
conv_transpose_modules = {2: nn.ConvTranspose2d}

flow_fold_heads = {2: 'B (heads dir) H W -> (B heads) H W dir'}
value_fold_heads = {2: 'B (heads C_i) H W -> (B heads) C_i H W'}
value_unfold_heads = {2: '(B heads) C_i H W -> B (heads C_i) H W'}
grid_samples = {2: custom_grid_sample2d}


# =============================================================================
# Flower Layers (verbatim -- base SelfWarp only, RK2/Conservative/Spherical
# variants not copied, not needed for this comparison round)
# =============================================================================

class SelfWarp(nn.Module):
    """Spatial warping operator: u(x) -> u(x + delta(x)). Powered by
    F.grid_sample (via custom_grid_sample2d) with periodic-boundary support.
    Multi-head: num_heads independent displacement fields, each warping its
    own slice of the value features (folded into the batch dim for a single
    vectorized grid_sample call)."""

    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads=32, padding_modes=["zeros", "zeros"], meta_dim=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.padding_modes = padding_modes
        self.num_heads = num_heads

        self.flow_head = nn.Sequential(
            conv_modules[self.n_spatial_dims](in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_modules[self.n_spatial_dims](out_channels, self.n_spatial_dims * self.num_heads, kernel_size=1)
        )
        self.value_head = conv_modules[self.n_spatial_dims](in_channels, out_channels, kernel_size=1)

        base_grid = create_grid(self.spatial_resolution)
        self.register_buffer("base_grid", base_grid)

    def flow(self, u, return_value=False):
        flow = self.flow_head(u)
        if return_value:
            value = self.value_head(u)
            return flow, value
        return flow

    def forward(self, u, meta=None):
        flow = self.flow_head(u)
        value = self.value_head(u)

        flow = rearrange(flow, flow_fold_heads[self.n_spatial_dims], dir=self.n_spatial_dims, heads=self.num_heads)

        base_grid_batch = self.base_grid.unsqueeze(0)
        grid = base_grid_batch + flow

        value = rearrange(value, value_fold_heads[self.n_spatial_dims], heads=self.num_heads)
        u_warp = grid_samples[self.n_spatial_dims](
            value,
            grid,
            mode='bilinear',
            padding_modes=self.padding_modes,
            align_corners=True
        )
        u_warp = rearrange(u_warp, value_unfold_heads[self.n_spatial_dims], heads=self.num_heads)
        return u_warp


# =============================================================================
# Core building block + U-Net assembly (verbatim)
# =============================================================================

class FlowerBlock(nn.Module):
    """Core building block of Flower: SelfWarp + residual + normalization (pre-norm)."""

    def __init__(self,
            in_channels,
            out_channels,
            spatial_resolution,
            num_heads,
            padding_modes,
            use_meta_conditioning,
            dropout_rate=0.0,
            groups=32,
            meta_dim=1,
            film_norm_type='group',
            drop_path_rate=0.0,
            use_checkpoint=False,
        ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.self_warp = SelfWarp(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            meta_dim=meta_dim
        )
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution
        self.id = conv_modules[self.n_spatial_dims](in_channels, out_channels, 1)
        self.use_meta_conditioning = use_meta_conditioning

        if use_meta_conditioning:
            self.norm = FiLM(
                num_channels=in_channels,
                meta_dim=meta_dim,
                norm_type=film_norm_type,
                num_groups=groups
            )
        else:
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels, affine=True)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def _forward_impl(self, x, meta):
        id_ = self.id(x)

        if self.use_meta_conditioning and meta is not None:
            x_normed = self.norm(x, meta)
        else:
            x_normed = self.norm(x)

        out = self.drop_path(self.self_warp(x_normed, meta)) + id_
        return self.dropout(self.act(out))

    def forward(self, x, meta=None):
        if self.use_checkpoint and self.training and x.requires_grad:
            return torch_checkpoint.checkpoint(
                self._forward_impl, x, meta, use_reentrant=False)
        return self._forward_impl(x, meta)


class DownsampleBlock(nn.Module):
    """Encoder block: FlowerBlock + strided convolution downsampling."""

    def __init__(
        self, in_channels, out_channels, spatial_resolution, num_heads, padding_modes,
        use_meta_conditioning=False, meta_dim=1, groups=32, dropout_rate=0.0,
        drop_path_rate=0.0, use_checkpoint=False,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.shift_layer = FlowerBlock(
            in_channels, in_channels, spatial_resolution=spatial_resolution, num_heads=num_heads,
            padding_modes=padding_modes, use_meta_conditioning=use_meta_conditioning,
            dropout_rate=dropout_rate, meta_dim=meta_dim, groups=groups,
            drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
        )
        self.downsample = conv_modules[self.n_spatial_dims](in_channels, out_channels, kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, meta=None):
        x_down = self.shift_layer(x, meta)
        x_down = self.act(self.downsample(x_down))
        return x, x_down


class UpsampleBlock(nn.Module):
    """Decoder block: FlowerBlock + transposed convolution upsampling."""

    def __init__(
        self, in_channels, skip_channels, out_channels, spatial_resolution, num_heads, padding_modes,
        use_meta_conditioning=False, meta_dim=1, groups=32, dropout_rate=0.0,
        drop_path_rate=0.0, use_checkpoint=False,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.fio = FlowerBlock(
            in_channels, out_channels - skip_channels, spatial_resolution=spatial_resolution,
            num_heads=num_heads, padding_modes=padding_modes, use_meta_conditioning=use_meta_conditioning,
            dropout_rate=dropout_rate, meta_dim=meta_dim, groups=groups,
            drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
        )

        self.upsample = conv_transpose_modules[self.n_spatial_dims](
            out_channels - skip_channels, out_channels - skip_channels, kernel_size=2, stride=2
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, meta=None):
        x = self.fio(x, meta)
        x = self.act(self.upsample(x))
        return x


class Flower(nn.Module):
    """Flower: a U-Net style neural operator with spatial warping layers.
    See flower_standalone.py's docstring for the full argument reference --
    copied verbatim, 2D-only (n_spatial_dims is accepted but this file only
    registers the 2D entries in conv_modules/grid_samples/etc. above)."""

    def __init__(
        self, dim_in, dim_out, n_spatial_dims, spatial_resolution,
        lifting_dim=160, n_levels=4, num_heads=40,
        boundary_condition_types=["PERIODIC"], dim_meta=0, groups=40,
        dropout_rate=0.0, bottleneck_depth=1, drop_path_rate=0.0, use_checkpoint=False,
    ):
        super().__init__()
        self.bottleneck_depth = bottleneck_depth

        self.input_dim = dim_in
        self.output_dim = dim_out
        self.spatial_resolution = list(spatial_resolution)
        self.lifting_dim = lifting_dim
        self.n_levels = n_levels
        self.dim_meta = dim_meta
        self.use_meta_conditioning = (dim_meta > 0)
        self.num_heads = num_heads
        self.boundary_condition_types = boundary_condition_types
        self.padding_type = derive_padding_strategies(self.boundary_condition_types, n_spatial_dims=n_spatial_dims)
        self.n_spatial_dims = n_spatial_dims
        self.groups = groups

        min_divisor = 2 ** (n_levels - 1)
        for i, d_i in enumerate(self.spatial_resolution):
            if d_i % min_divisor != 0:
                raise ValueError(
                    f"Dimension {i} must be divisible by 2^(n_levels-1) = {min_divisor}, "
                    f"or n_levels={n_levels}, but is {d_i}."
                )

        coord_dim = self.n_spatial_dims
        if self.use_meta_conditioning:
            coord_dim += dim_meta

        self.lift = conv_modules[self.n_spatial_dims](dim_in + coord_dim, lifting_dim, kernel_size=1)

        channel_multipliers = [2.0 ** i for i in range(n_levels)]
        self.encoder_channels = [int(lifting_dim * mult) for mult in channel_multipliers]

        total_depth = 2 * (n_levels - 1) + bottleneck_depth
        if total_depth > 1:
            dpr = [drop_path_rate * i / (total_depth - 1) for i in range(total_depth)]
        else:
            dpr = [0.0] * total_depth
        dpr_idx = 0

        self.encoder_blocks = nn.ModuleList()
        current_spatial_resolution = self.spatial_resolution

        for i in range(n_levels - 1):
            in_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i + 1]
            self.encoder_blocks.append(
                DownsampleBlock(
                    in_ch, out_ch, current_spatial_resolution, num_heads, self.padding_type,
                    self.use_meta_conditioning, meta_dim=dim_meta, groups=self.groups,
                    dropout_rate=dropout_rate, drop_path_rate=dpr[dpr_idx], use_checkpoint=use_checkpoint,
                )
            )
            dpr_idx += 1
            current_spatial_resolution = [d_i // 2 for d_i in current_spatial_resolution]

        bottleneck_channels = self.encoder_channels[-1]
        self.bottleneck = nn.ModuleList()
        for _ in range(bottleneck_depth):
            self.bottleneck.append(
                FlowerBlock(
                    bottleneck_channels, bottleneck_channels, spatial_resolution=current_spatial_resolution,
                    num_heads=num_heads, padding_modes=self.padding_type,
                    use_meta_conditioning=self.use_meta_conditioning, meta_dim=dim_meta, groups=groups,
                    dropout_rate=dropout_rate, drop_path_rate=dpr[dpr_idx], use_checkpoint=use_checkpoint,
                )
            )
            dpr_idx += 1

        self.decoder_blocks = nn.ModuleList()
        decoder_in_ch = bottleneck_channels

        for i in range(n_levels - 1, 0, -1):
            skip_ch = self.encoder_channels[i - 1]
            out_ch = skip_ch * 2
            self.decoder_blocks.append(
                UpsampleBlock(
                    decoder_in_ch, skip_ch, out_ch, current_spatial_resolution, num_heads, self.padding_type,
                    self.use_meta_conditioning, meta_dim=dim_meta, groups=self.groups,
                    dropout_rate=dropout_rate, drop_path_rate=dpr[dpr_idx], use_checkpoint=use_checkpoint,
                )
            )
            dpr_idx += 1
            decoder_in_ch = out_ch
            current_spatial_resolution = [2 * d_i for d_i in current_spatial_resolution]

        self.project = nn.Sequential(
            conv_modules[self.n_spatial_dims](out_ch, lifting_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_modules[self.n_spatial_dims](lifting_dim, dim_out, kernel_size=1),
        )
        self.dropout = nn.Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)

        coord_grid = rearrange(create_grid(self.spatial_resolution), "... xy -> 1 xy ...").contiguous()
        self.register_buffer("coord_grid", coord_grid)

        self.learned_pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(
                1, lifting_dim,
                *[self.spatial_resolution[j] if j == i else 1 for j in range(len(self.spatial_resolution))]
            ))
            for i in range(len(self.spatial_resolution))
        ])

    def forward(self, x, meta=None):
        batchsize = x.shape[0]

        expand_shape = (batchsize, -1) + (-1,) * self.n_spatial_dims
        grid = self.coord_grid.expand(*expand_shape)

        if self.use_meta_conditioning and meta is not None:
            view_shape = (batchsize, self.dim_meta) + (1,) * self.n_spatial_dims
            expand_shape = (-1, -1) + tuple(self.spatial_resolution)
            meta_broadcast = meta.view(*view_shape).expand(*expand_shape)
            grid = torch.cat((grid, meta_broadcast), dim=1)

        x = torch.cat((x, grid), dim=1)

        x = self.lift(x)
        for axis_embed in self.learned_pos_embed:
            x = x + axis_embed

        skip_connections = []
        for encoder_block in self.encoder_blocks:
            skip, x = encoder_block(x, meta)
            skip_connections.append(skip)

        for bottleneck_block in self.bottleneck:
            x = bottleneck_block(x, meta)

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder_block(x, meta)
            x = torch.cat([x, skip], dim=1)

        x = self.dropout(x)
        prediction = self.project(x)
        return prediction
