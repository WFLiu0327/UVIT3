"""UVIT3 segmentation model using the official ViT^3 TTT block."""

from __future__ import annotations

from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from ttt_block import TTT

def _groups(channels: int, max_groups: int = 8) -> int:
    """Choose a GroupNorm group count that divides ``channels``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    """Two-convolution residual block used by the shallow encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class LocalGlobalViT3Block(nn.Module):
    """Dense-prediction block combining local depthwise conv and official ViT^3."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        use_local_branch: bool = True,
        fusion: str = "gated",
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if fusion not in {"add", "gated"}:
            raise ValueError("fusion must be 'add' or 'gated'")

        self.dim = dim
        self.use_local_branch = use_local_branch
        self.fusion = fusion
        self.norm1 = nn.LayerNorm(dim)
        self.ttt = TTT(dim=dim, num_heads=num_heads)

        if use_local_branch:
            self.local = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.GroupNorm(_groups(dim), dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(dim, dim, 1, bias=False),
            )
            self.gate = (
                nn.Conv2d(dim * 2, dim, 1)
                if fusion == "gated"
                else nn.Identity()
            )
        else:
            self.local = None
            self.gate = None

        hidden = max(dim, int(dim * mlp_ratio))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), h, w

    @staticmethod
    def _to_map(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = tokens.shape
        if n != h * w:
            raise RuntimeError(f"token count {n} does not match spatial shape {(h, w)}")
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, h, w = self._to_tokens(x)
        ttt_features = self.ttt(self.norm1(tokens), h=h, w=w)
        ttt_features = self._to_map(ttt_features, h, w)

        if not self.use_local_branch:
            fused = ttt_features
        else:
            local_features = self.local(x)
            if self.fusion == "add":
                fused = local_features + ttt_features
            else:
                gate = torch.sigmoid(self.gate(torch.cat([local_features, ttt_features], dim=1)))
                fused = gate * ttt_features + (1.0 - gate) * local_features

        x = x + fused
        mlp_tokens = self.norm2(x.flatten(2).transpose(1, 2))
        x = x + self.mlp(self._to_map(mlp_tokens, h, w))
        return x


class TTTStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, *, use_local_branch: bool, fusion: str) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                LocalGlobalViT3Block(
                    dim,
                    heads,
                    use_local_branch=use_local_branch,
                    fusion=fusion,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CrossScaleViT3Memory(nn.Module):
    """Build bottleneck memory from 1/4, 1/8 and 1/16 encoder features."""

    def __init__(
        self,
        shallow_channels: int,
        middle_channels: int,
        deep_channels: int,
        memory_channels: int,
        heads: int,
        *,
        depth: int,
    ) -> None:
        super().__init__()
        self.shallow_down = nn.Sequential(
            ConvNormAct(shallow_channels, middle_channels, stride=2),
            ConvNormAct(middle_channels, memory_channels, stride=2),
        )
        self.middle_down = ConvNormAct(middle_channels, memory_channels, stride=2)
        self.deep_projection = nn.Sequential(
            nn.Conv2d(deep_channels, memory_channels, 1, bias=False),
            nn.GroupNorm(_groups(memory_channels), memory_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(memory_channels * 3, memory_channels, 1, bias=False),
            nn.GroupNorm(_groups(memory_channels), memory_channels),
            nn.SiLU(inplace=True),
            ConvBlock(memory_channels, memory_channels),
        )
        self.memory = TTTStage(
            memory_channels,
            depth,
            heads,
            use_local_branch=True,
            fusion="gated",
        )
        self.refine = ConvBlock(memory_channels, memory_channels)

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(
            x, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        shallow: torch.Tensor,
        middle: torch.Tensor,
        deep: torch.Tensor,
    ) -> torch.Tensor:
        shallow = self._match_size(self.shallow_down(shallow), deep)
        middle = self._match_size(self.middle_down(middle), deep)
        deep_projected = self.deep_projection(deep)
        fused = self.fuse(torch.cat((shallow, middle, deep_projected), dim=1))
        return self.refine(deep_projected + self.memory(fused))


class SemanticPyramidDecoderStage(nn.Module):
    """Fuse local skip, decoder state, and global ViT3 memory without gating."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        memory_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = ConvNormAct(in_channels, out_channels)
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.memory_projection = nn.Sequential(
            nn.Conv2d(memory_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = ConvBlock(out_channels * 3, out_channels)
        self.refine = ConvBlock(out_channels, out_channels)

    def forward(
        self,
        decoder: torch.Tensor,
        skip: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        target_size = skip.shape[-2:]
        decoder = F.interpolate(
            decoder, size=target_size, mode="bilinear", align_corners=False
        )
        decoder = self.up(decoder)
        
        skip = self.skip_projection(skip)
        
        memory = self.memory_projection(memory)
        memory = F.interpolate(
            memory, size=target_size, mode="bilinear", align_corners=False
        )
        return self.refine(self.fuse(torch.cat((decoder, skip, memory), dim=1)))


class MultiScalePredictionHead(nn.Module):
    """Fuse decoder predictions while returning one protocol-compatible logits map."""

    def __init__(self, channels: Sequence[int], num_classes: int) -> None:
        super().__init__()
        self.predictions = nn.ModuleList(
            [nn.Conv2d(channel, num_classes, 1) for channel in channels]
        )
        self.fuse = nn.Conv2d(num_classes * len(channels), num_classes, 1, bias=True)
        self._initialize_fusion(num_classes, len(channels))

    def _initialize_fusion(self, num_classes: int, levels: int) -> None:
        level_weights = torch.tensor(
            (0.05, 0.10, 0.20, 0.65), dtype=self.fuse.weight.dtype
        )
        if levels != len(level_weights):
            level_weights = torch.full(
                (levels,), 1.0 / levels, dtype=self.fuse.weight.dtype
            )
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            for class_index in range(num_classes):
                for level_index, weight in enumerate(level_weights):
                    input_index = level_index * num_classes + class_index
                    self.fuse.weight[class_index, input_index, 0, 0] = weight

    def forward(
        self, features: Sequence[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        if len(features) != len(self.predictions):
            raise ValueError("feature count does not match prediction heads")
        logits = [
            F.interpolate(
                head(feature), size=output_size, mode="bilinear", align_corners=False
            )
            for head, feature in zip(self.predictions, features)
        ]
        return self.fuse(torch.cat(logits, dim=1))


class UVIT3(nn.Module):
    """U-shape segmentation network centered on cross-scale ViT3 memory."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: Sequence[int] = (64, 128, 256, 384),
        depths: Sequence[int] = (2, 2, 4, 2),
        num_heads: Sequence[int] = (2, 4, 8, 12),
        *,
        detail_channels: int = 32,
        memory_depth: int = 1,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(depths) == len(num_heads) == 4):
            raise ValueError(
                "channels, depths and num_heads must each contain four entries"
            )
        if min(*channels, *depths, *num_heads, detail_channels, memory_depth) < 1:
            raise ValueError("channel, depth and head values must be positive")
        if any(channel % heads != 0 for channel, heads in zip(channels, num_heads)):
            raise ValueError("each channel width must be divisible by its head count")

        c0, c1, c2, c3 = channels
        self.config = {
            "architecture": "UVIT3",
            "channels": tuple(channels),
            "depths": tuple(depths),
            "num_heads": tuple(num_heads),
            "detail_channels": detail_channels,
            "memory_depth": memory_depth,
            "memory_channels": c2,
            "memory_scales": ("1/4", "1/8", "1/16"),
            "decoder_fusion": "skip+decoder+vit3_memory_concat",
        }

        self.detail_stem = nn.Sequential(
            ConvNormAct(in_channels, detail_channels),
            ConvBlock(detail_channels, detail_channels),
        )
        self.stem = nn.Sequential(
            ConvNormAct(detail_channels, c0, stride=2),
            ConvBlock(c0, c0),
        )
        self.stage1 = nn.Sequential(*[ConvBlock(c0, c0) for _ in range(depths[0])])

        self.down1 = ConvNormAct(c0, c1, stride=2)
        self.stage2 = TTTStage(
            c1, depths[1], num_heads[1], use_local_branch=True, fusion="gated"
        )
        self.down2 = ConvNormAct(c1, c2, stride=2)
        self.stage3 = TTTStage(
            c2, depths[2], num_heads[2], use_local_branch=True, fusion="gated"
        )
        self.down3 = ConvNormAct(c2, c3, stride=2)
        self.stage4 = TTTStage(
            c3, depths[3], num_heads[3], use_local_branch=True, fusion="gated"
        )

        self.memory = CrossScaleViT3Memory(
            c1,
            c2,
            c3,
            c2,
            num_heads[2],
            depth=memory_depth,
        )
        self.decoder3 = SemanticPyramidDecoderStage(c2, c2, c2, c2)
        self.decoder2 = SemanticPyramidDecoderStage(c2, c1, c2, c1)
        self.decoder1 = SemanticPyramidDecoderStage(c1, c0, c2, c0)
        self.decoder0 = SemanticPyramidDecoderStage(
            c0, detail_channels, c2, detail_channels
        )
        self.head = MultiScalePredictionHead(
            (c2, c1, c0, detail_channels),
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        detail = self.detail_stem(x)
        e1 = self.stage1(self.stem(detail))
        e2 = self.stage2(self.down1(e1))
        e3 = self.stage3(self.down2(e2))
        e4 = self.stage4(self.down3(e3))

        memory = self.memory(e2, e3, e4)
        d3 = self.decoder3(memory, e3, memory)
        d2 = self.decoder2(d3, e2, memory)
        d1 = self.decoder1(d2, e1, memory)
        d0 = self.decoder0(d1, detail, memory)
        return self.head((d3, d2, d1, d0), output_size)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_uvit3(**kwargs: object) -> UVIT3:
    return UVIT3(**kwargs)


__all__ = [
    "CrossScaleViT3Memory",
    "MultiScalePredictionHead",
    "SemanticPyramidDecoderStage",
    "UVIT3",
    "build_uvit3",
]
