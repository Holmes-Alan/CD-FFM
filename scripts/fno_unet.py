"""
fno_unet.py
===========
Gives FNO the SAME U-Net scaffold as the real Flower (flower_layers.py):
same channel-doubling per level, same lift -> coordinate-grid-concat ->
learned axis positional embedding treatment, same skip-connection pattern,
same strided-conv down/up-sampling. Only the per-level mixing primitive
differs (FNOBlock's spectral conv vs. FlowerBlock's SelfWarp) -- so the
backbone comparison isolates the mixing mechanism, not flat-vs-U-Net
architecture style too.

Deliberately a SEPARATE, parallel implementation rather than a modified
flower_layers.py -- that file stays an unmodified, verbatim copy of the
real Flower code (per instruction); this mirrors its assembly pattern
without touching it.
"""
import torch
import torch.nn as nn

from fno_layers import SpectralConv2d


def _capped_modes(modes1, modes2, spatial_resolution):
    """SpectralConv2d allocates its weight at the FULL (modes1,modes2) size
    regardless of what forward()'s runtime clamp (min(modes, actual H/Wf))
    ends up reading -- so passing the same top-level modes1/modes2 to every
    U-Net level, where channels grow to 512 at the bottleneck, allocates
    ~100M dead (never-gradient-touched) parameters there alone. Capping at
    CONSTRUCTION time, per level, to what that level's resolution can
    actually support removes the waste instead of relying on forward()'s
    clamp to just avoid an index-out-of-range crash."""
    H, W = spatial_resolution
    return min(modes1, H), min(modes2, W // 2 + 1)


class FNOBlock(nn.Module):
    """Mirrors FlowerBlock's role (pre-norm + mixer + residual + GELU) with
    FNO's spectral channel-mixing in place of SelfWarp."""

    def __init__(self, in_channels, out_channels, spatial_resolution, modes1=12, modes2=16, groups=32):
        super().__init__()
        modes1, modes2 = _capped_modes(modes1, modes2, spatial_resolution)
        self.spectral = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.bypass = nn.Conv2d(in_channels, out_channels, 1)
        self.id = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        id_ = self.id(x)
        x_normed = self.norm(x)
        out = (self.spectral(x_normed) + self.bypass(x_normed)) + id_
        return self.act(out)


class FNODownsampleBlock(nn.Module):
    """Mirrors flower_layers.DownsampleBlock: FNOBlock + strided-conv downsample."""

    def __init__(self, in_channels, out_channels, spatial_resolution, modes1=12, modes2=16, groups=32):
        super().__init__()
        self.shift_layer = FNOBlock(in_channels, in_channels, spatial_resolution, modes1, modes2, groups)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.shift_layer(x)
        x_down = self.act(self.downsample(x_down))
        return x, x_down


class FNOUpsampleBlock(nn.Module):
    """Mirrors flower_layers.UpsampleBlock: FNOBlock + transposed-conv upsample."""

    def __init__(self, in_channels, skip_channels, out_channels, spatial_resolution, modes1=12, modes2=16, groups=32):
        super().__init__()
        self.fio = FNOBlock(in_channels, out_channels - skip_channels, spatial_resolution, modes1, modes2, groups)
        self.upsample = nn.ConvTranspose2d(
            out_channels - skip_channels, out_channels - skip_channels, kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fio(x)
        x = self.act(self.upsample(x))
        return x


def _create_grid(spatial_resolution):
    """Same convention as flower_layers.create_grid: (H,W,2) in [-1,1]."""
    meshgrid = torch.meshgrid(*[torch.linspace(-1., 1., d) for d in spatial_resolution], indexing="ij")
    return torch.stack(meshgrid[::-1], dim=-1)


class FNOUNet(nn.Module):
    """Mirrors flower_layers.Flower's assembly exactly: dim_in -> dim_out
    over spatial_resolution, n_levels U-Net stages, channel doubling per
    level, coordinate-grid-concat + learned per-axis positional embedding,
    skip connections -- only the per-level block differs (FNOBlock vs
    FlowerBlock). Same divisibility requirement (spatial dims must be
    divisible by 2**(n_levels-1)) for the same reason: strided conv/
    transposed-conv halving needs even sizes at every level."""

    def __init__(self, dim_in, dim_out, spatial_resolution, lifting_dim=64, n_levels=4,
                 modes1=12, modes2=16, groups=8, bottleneck_depth=1, mode_reference_resolution=None):
        """mode_reference_resolution: decouples _capped_modes's cap from THIS
        model's own spatial_resolution -- defaults to spatial_resolution
        (unchanged behavior) but can be set to a DIFFERENT resolution's
        halving hierarchy so two FNOUNets built at genuinely different
        spatial_resolution (e.g. an ERA5-grid encoder vs. a patch-pretrain
        model at NGCD resolution) end up with IDENTICAL SpectralConv2d
        weight shapes -- weight shape is (in_ch,out_ch,modes1,modes2),
        determined entirely by the modes passed in, not by the actual
        forward-time resolution (that only affects a runtime clamp) -- so
        as long as the reference hierarchy's cap at every level is <= both
        models' actual resolution at that level (true here: the patch
        model's resolution is always larger than the ERA5 encoder's own),
        this is a safe, lossless way to force shared weight shapes for
        cross-model parameter transfer (see train_patch_pretrain.py)."""
        super().__init__()
        self.spatial_resolution = list(spatial_resolution)
        self.n_levels = n_levels

        min_divisor = 2 ** (n_levels - 1)
        for i, d_i in enumerate(self.spatial_resolution):
            if d_i % min_divisor != 0:
                raise ValueError(f"Dimension {i} must be divisible by 2**(n_levels-1)={min_divisor}, got {d_i}")

        coord_dim = 2  # no meta-conditioning in this project, same as flower_layers' dim_meta=0 case
        self.lift = nn.Conv2d(dim_in + coord_dim, lifting_dim, kernel_size=1)

        channel_multipliers = [2 ** i for i in range(n_levels)]
        self.encoder_channels = [lifting_dim * m for m in channel_multipliers]

        self.encoder_blocks = nn.ModuleList()
        current_mode_ref = list(mode_reference_resolution) if mode_reference_resolution else self.spatial_resolution
        for i in range(n_levels - 1):
            in_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i + 1]
            self.encoder_blocks.append(
                FNODownsampleBlock(in_ch, out_ch, current_mode_ref, modes1, modes2, groups))
            current_mode_ref = [d // 2 for d in current_mode_ref]

        bottleneck_channels = self.encoder_channels[-1]
        self.bottleneck = nn.ModuleList([
            FNOBlock(bottleneck_channels, bottleneck_channels, current_mode_ref, modes1, modes2, groups)
            for _ in range(bottleneck_depth)
        ])

        self.decoder_blocks = nn.ModuleList()
        decoder_in_ch = bottleneck_channels
        out_ch = bottleneck_channels  # overwritten in the loop when n_levels > 1
        for i in range(n_levels - 1, 0, -1):
            skip_ch = self.encoder_channels[i - 1]
            out_ch = skip_ch * 2
            self.decoder_blocks.append(
                FNOUpsampleBlock(decoder_in_ch, skip_ch, out_ch, current_mode_ref, modes1, modes2, groups))
            decoder_in_ch = out_ch
            current_mode_ref = [2 * d for d in current_mode_ref]

        self.project = nn.Sequential(
            nn.Conv2d(out_ch, lifting_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(lifting_dim, dim_out, kernel_size=1),
        )

        coord_grid = _create_grid(self.spatial_resolution).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,2,H,W)
        self.register_buffer("coord_grid", coord_grid)
        self.learned_pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, lifting_dim, *[self.spatial_resolution[j] if j == i else 1 for j in range(2)]))
            for i in range(2)
        ])

    def forward(self, x):
        B = x.shape[0]
        grid = self.coord_grid.expand(B, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)
        x = self.lift(x)
        for axis_embed in self.learned_pos_embed:
            x = x + axis_embed

        skip_connections = []
        for block in self.encoder_blocks:
            skip, x = block(x)
            skip_connections.append(skip)

        for block in self.bottleneck:
            x = block(x)

        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = block(x)
            x = torch.cat([x, skip], dim=1)

        return self.project(x)
