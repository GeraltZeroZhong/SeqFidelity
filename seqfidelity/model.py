"""Bidirectional paired sequence scoring with explicit adaptive routing.

The contextual encoder, cosine boundary router, hard selection, probability-
weighted EMA dechunking, residual path and decoder are inspired by H-Net:
https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py and
https://github.com/goombalab/hnet/blob/main/hnet/models/hnet.py.
This implementation uses a fixed top-K budget, bidirectional convolution and
attention, pair routing, and explicit intervention state.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    mode: str = 'learned'
    dim: int = 256
    depth: int = 4
    compression: int = 4
    heads: int = 8
    encoder_depth: int = 2
    decoder_depth: int = 1
    expansion: int = 4
    kernel_size: int = 7
    dropout: float = 0.0
    cnn_dim: int = 96
    cnn_depth: int = 5
    architecture: str = 'routed'
    external_skip: bool = True
    branch_fusion: str = 'sum'

    def __post_init__(self):
        if min(self.dim, self.heads, self.cnn_dim, self.expansion, self.kernel_size) < 1:
            raise ValueError('Feature sizes, attention heads, expansion, and kernel_size must be positive')
        if min(self.encoder_depth, self.decoder_depth, self.cnn_depth) < 0:
            raise ValueError('Encoder, decoder, and CNN depths must be nonnegative')
        if not 0 <= self.dropout <= 1:
            raise ValueError('dropout must lie between zero and one')
        if self.architecture not in {'routed', 'legnet'}:
            raise ValueError(f'Unknown architecture {self.architecture!r}')
        modes = {'none', 'fixed', 'aggregate', 'learned', 'shared', 'coupled', 'no_backbone', 'cnn'}
        if self.mode not in modes:
            raise ValueError(f'Unknown mode {self.mode!r}')
        if self.dim % self.heads:
            raise ValueError('dim must be divisible by heads')
        if self.compression < 1 or self.depth < 0:
            raise ValueError('compression must be positive and depth nonnegative')
        if self.kernel_size % 2 != 1:
            raise ValueError('Convolution kernel_size must be odd')
        if self.branch_fusion not in {'sum', 'rms_balanced'}:
            raise ValueError(f'Unknown branch fusion {self.branch_fusion!r}')


@dataclass
class RouteState:
    """Complete sequence-dependent routing state, without hidden features.

    Coordinates are original zero-based nucleotide positions. dechunk_map maps
    each input position to its preceding selected boundary. Padded indices have
    no meaning and are masked by selected_mask. Rebuild all derived fields with
    route_from_probs after changing boundary positions or probabilities.
    """
    indices: Tensor
    mask: Tensor
    probs: Tensor
    selected_probs: Tensor
    dechunk_map: Tensor
    coords: Tensor
    lengths: Tensor
    selected_lengths: Tensor
    valid_mask: Tensor
    selected_mask: Tensor

    def detached(self):
        return RouteState(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def to(self, device):
        return RouteState(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def split(self, batch):
        return tuple(RouteState(**{f.name: getattr(self, f.name)[s] for f in fields(self)})
                     for s in (slice(None, batch), slice(batch, None)))

    @staticmethod
    def concatenate(routes):
        return RouteState(**{f.name: torch.cat([getattr(r, f.name) for r in routes], dim=0)
                             for f in fields(RouteState)})


def length_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def masked_mean(x, mask):
    return (x.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def _position_encoding(width, dim, device, dtype):
    pos = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                     * (-math.log(10000.0) / dim))
    phase = pos * freq
    out = torch.zeros(width, dim, device=device, dtype=torch.float32)
    out[:, 0::2] = phase.sin()
    out[:, 1::2] = phase[:, :dim // 2].cos()
    return out.to(dtype)


class ConvBlock(nn.Module):
    def __init__(self, dim, expansion, kernel, dropout, dilation=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel // 2),
                              groups=dim, dilation=dilation)
        self.mix = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, expansion * dim), nn.GELU(),
                                nn.Linear(expansion * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        m = mask[..., None]
        z = self.norm1(x) * m
        z = self.conv(z.transpose(1, 2)).transpose(1, 2)
        x = (x + self.drop(self.mix(F.gelu(z)))) * m
        return (x + self.drop(self.ff(self.norm2(x)))) * m


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, config.expansion * d), nn.GELU(),
                                nn.Linear(config.expansion * d, d))
        self.drop = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, mask):
        batch, width, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, width, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        z = z.transpose(1, 2).reshape(batch, width, dim)
        x = (x + self.drop(self.out(z))) * mask[..., None]
        return (x + self.drop(self.ff(self.norm2(x)))) * mask[..., None]


class BoundaryRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)

    def forward(self, encoded, lengths):
        q = F.normalize(self.q_proj(encoded[:, :-1]).float(), dim=-1)
        k = F.normalize(self.k_proj(encoded[:, 1:]).float(), dim=-1)
        probs = ((1.0 - (q * k).sum(-1)) * 0.5).clamp(0, 1)
        probs = F.pad(probs, (1, 0), value=1.0)
        return probs * length_mask(lengths, encoded.shape[1])


class PairedSequenceModel(nn.Module):
    """A shared architecture with independently configurable routing.

    shared: 0.5*(p_ref+p_alt). coupled: p'_self=0.75*p_self+0.25*p_partner.
    Both are pair-aware and exchange equivariant. Single-sequence evaluation in
    these modes is equivalent to pairing that sequence with itself.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        d = c.cnn_dim if c.mode == 'cnn' else c.dim
        self.feature_dim = d
        self.embedding = nn.Embedding(5, d)
        self.encoder = nn.ModuleList([
            ConvBlock(d, c.expansion, c.kernel_size, c.dropout, 2 ** i if c.mode == 'cnn' else 1)
            for i in range(c.cnn_depth if c.mode == 'cnn' else c.encoder_depth)])
        self.router = BoundaryRouter(d)
        self.backbone = nn.ModuleList([] if c.mode in {'no_backbone', 'cnn'}
                                      else [AttentionBlock(c) for _ in range(c.depth)])
        self.residual_proj = nn.Linear(d, d)
        self.decoder = nn.ModuleList([] if c.mode == 'cnn' else [
            ConvBlock(d, c.expansion, c.kernel_size, c.dropout) for _ in range(c.decoder_depth)])
        self.readout = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # Initialize shared modules before applying deterministic branch balancing.
        if c.branch_fusion == 'rms_balanced':
            nn.init.eye_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def encode(self, tokens, lengths):
        mask = length_mask(lengths, tokens.shape[1])
        x = self.embedding(tokens)
        x = (x + _position_encoding(x.shape[1], x.shape[2], x.device, x.dtype)) * mask[..., None]
        for block in self.encoder:
            x = block(x, mask)
        return x

    def route_from_probs(self, probs, lengths, indices=None):
        """Rebuild all state; optional indices force a hard intervention.

        Custom indices must start at zero and contain ceil(length/N) valid,
        sorted coordinates per row. Padded columns are ignored.
        """
        batch, width = probs.shape
        valid = length_mask(lengths, width)
        uncompressed = self.config.mode in {'none', 'no_backbone', 'cnn'}
        compression = 1 if uncompressed else self.config.compression
        counts = (lengths + compression - 1) // compression
        kmax = (width + compression - 1) // compression
        selected_mask = length_mask(counts, kmax)
        probs = probs * valid
        if indices is None:
            if uncompressed or self.config.mode in {'fixed', 'aggregate'}:
                indices = torch.arange(kmax, device=probs.device)[None].expand(batch, -1) * compression
                indices = indices.clamp_max(width - 1)
            else:
                scores = probs.masked_fill(~valid, -torch.inf)
                scores = scores.clone()
                scores[:, 0] = torch.inf
                rank_indices = scores.topk(kmax, dim=1, sorted=True).indices
                indices = rank_indices.masked_fill(~selected_mask, width).sort(dim=1).values.clamp_max(width - 1)
        if indices.shape != (batch, kmax):
            raise ValueError(f'Expected indices shape {(batch, kmax)}, received {tuple(indices.shape)}')
        hard = torch.zeros(batch, width, dtype=torch.long, device=probs.device)
        hard.scatter_add_(1, indices, selected_mask.long())
        mask = hard.bool() & valid
        selected_probs = probs.gather(1, indices) * selected_mask
        dechunk_map = (mask.long().cumsum(1) - 1).clamp_min(0)
        return RouteState(indices, mask, probs, selected_probs, dechunk_map,
                          indices.clone(), lengths, counts, valid, selected_mask)

    def make_route(self, encoded, lengths):
        if self.config.mode in {'none', 'no_backbone', 'cnn'}:
            probs = length_mask(lengths, encoded.shape[1]).float()
        elif self.config.mode in {'fixed', 'aggregate'}:
            pos = torch.arange(encoded.shape[1], device=encoded.device)
            probs = ((pos % self.config.compression) == 0).float()[None].expand(encoded.shape[0], -1)
        else:
            probs = self.router(encoded, lengths)
        return self.route_from_probs(probs, lengths)

    def _dechunk(self, compressed, route, ema_enabled=True):
        if not ema_enabled or self.config.mode in {'none', 'fixed', 'aggregate', 'no_backbone', 'cnn'}:
            smoothed = compressed
        else:
            # Parallel exact EMA: z_i = p_i*v_i + (1-p_i)*z_(i-1).
            # Log-space weights avoid tiny-cumprod division and token CUDA loops.
            p = route.selected_probs.float().clamp(1e-4, 1 - 1e-4)
            p = torch.cat((torch.ones_like(p[:, :1]), p[:, 1:]), dim=1)
            p = p * route.selected_mask
            cumulative = F.pad(torch.log1p(-p[:, 1:]).cumsum(1), (1, 0))
            log_weights = cumulative[:, :, None] - cumulative[:, None, :]
            k = compressed.shape[1]
            lower = torch.ones(k, k, device=compressed.device, dtype=torch.bool).tril()
            weights = log_weights.masked_fill(~lower, -torch.inf).exp() * p[:, None, :]
            with torch.autocast(device_type=compressed.device.type, enabled=False):
                smoothed = torch.bmm(weights.float(), compressed.float())
            smoothed = smoothed.to(compressed.dtype)
        x = smoothed.gather(1, route.dechunk_map[..., None].expand(-1, -1, compressed.shape[-1]))
        # Official-style unit-valued straight-through confidence gate gives
        # gradients also at unselected sites, without changing the forward value.
        confidence = torch.where(route.mask, route.probs, 1 - route.probs)
        straight_through = 1 + (confidence - confidence.detach())
        return x * straight_through[..., None] * route.valid_mask[..., None]

    def fuse_branches(self, dechunked, bypass, bypass_active=None):
        """Sum the paths, or normalize their RMS with an identity-initialized bypass."""
        if self.config.branch_fusion == 'sum':
            return dechunked + bypass
        active = self.config.external_skip if bypass_active is None else bypass_active
        with torch.autocast(device_type=dechunked.device.type, enabled=False):
            def rms_normalize(value):
                value = value.float()
                return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            x = rms_normalize(dechunked)
            if active:
                x = (x + rms_normalize(bypass)) / math.sqrt(2.0)
        return x.to(dechunked.dtype)

    def decode_branches(self, dechunked, bypass, lengths, return_features=False,
                        bypass_active=None):
        """Fuse full-resolution paths, then apply the existing decoder and head."""
        mask = length_mask(lengths, dechunked.shape[1])
        x = self.fuse_branches(dechunked, bypass, bypass_active=bypass_active)
        for block in self.decoder:
            x = block(x, mask)
        x = x * mask[..., None]
        with torch.autocast(device_type=x.device.type, enabled=False):
            prediction = self.readout(masked_mean(x, mask).float()).squeeze(-1)
        return (prediction, x) if return_features else prediction

    def forward_with_route(self, encoded, lengths, route, zero_residual=False, return_features=False,
                           ema_enabled=True):
        """Re-select from recipient content; residual also stays with recipient."""
        d = encoded.shape[-1]
        if self.config.mode in {'no_backbone', 'cnn'}:
            selected = encoded
            compressed = torch.zeros_like(encoded)
            dechunked = compressed
        else:
            if self.config.mode == 'aggregate':
                # Fixed intervals [j*N, min((j+1)*N, L)); padding contributes
                # neither values nor denominator. Coordinates remain j*N.
                n = self.config.compression
                k = route.indices.shape[1]
                pad = k * n - encoded.shape[1]
                valid = route.valid_mask
                values = F.pad(encoded * valid[..., None], (0, 0, 0, pad))
                counts = F.pad(valid, (0, pad)).reshape(len(encoded), k, n).sum(2)
                selected = values.float().reshape(len(encoded), k, n, d).sum(2)
                selected = (selected / counts.clamp_min(1)[..., None]).to(encoded.dtype)
            else:
                selected = encoded.gather(1, route.indices[..., None].expand(-1, -1, d))
            selected = selected * route.selected_mask[..., None]
            compressed = selected
            for block in self.backbone:
                compressed = block(compressed, route.selected_mask)
            dechunked = self._dechunk(compressed, route, ema_enabled=ema_enabled)
        bypass = (self.residual_proj(encoded) if self.config.external_skip and not zero_residual
                  else torch.zeros_like(encoded))
        prediction, x = self.decode_branches(
            dechunked, bypass, lengths, return_features=True,
            bypass_active=self.config.external_skip and not zero_residual)
        if not return_features:
            return prediction
        # In the uncompressed mode, selected states equal encoder states.
        if self.config.mode == 'none':
            selected, dechunked = encoded, compressed
        result = {'prediction': prediction, 'full': x, 'backbone': compressed,
                  'encoder': encoded, 'compressed': selected, 'dechunked': dechunked,
                  'bypass': bypass, 'lengths': lengths}
        coords = torch.arange(encoded.shape[1], device=encoded.device)[None].expand(len(encoded), -1)
        for stage in ('encoder', 'compressed', 'backbone', 'dechunked', 'full', 'bypass'):
            reduced = stage in {'compressed', 'backbone'}
            result[f'{stage}_coords'] = route.coords if reduced else coords
            result[f'{stage}_lengths'] = route.selected_lengths if reduced else lengths
        return result

    def forward_pair(self, ref, alt, lengths, return_features=False, zero_residual=False):
        batch = ref.shape[0]
        both_lengths = torch.cat((lengths, lengths))
        encoded = self.encode(torch.cat((ref, alt)), both_lengths)
        if self.config.mode in {'shared', 'coupled'}:
            p_ref, p_alt = self.router(encoded, both_lengths).split(batch)
            if self.config.mode == 'shared':
                p_ref = p_alt = (p_ref + p_alt) * 0.5
            else:
                p_ref, p_alt = 0.75 * p_ref + 0.25 * p_alt, 0.75 * p_alt + 0.25 * p_ref
            route = self.route_from_probs(torch.cat((p_ref, p_alt)), both_lengths)
        else:
            route = self.make_route(encoded, both_lengths)
        result = self.forward_with_route(encoded, both_lengths, route, zero_residual, return_features)
        predictions = result['prediction'] if return_features else result
        y_ref, y_alt = predictions.split(batch)
        route_ref, route_alt = route.split(batch)
        output = {'delta': y_alt.float() - y_ref.float(), 'y_ref': y_ref, 'y_alt': y_alt,
                  'route_ref': route_ref, 'route_alt': route_alt}
        if return_features:
            output['features_ref'] = {name: value[:batch] for name, value in result.items()}
            output['features_alt'] = {name: value[batch:] for name, value in result.items()}
        return output

    def forward(self, tokens, lengths):
        encoded = self.encode(tokens, lengths)
        return self.forward_with_route(encoded, lengths, self.make_route(encoded, lengths))

    def parameter_count(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def freeze_except_router(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.router.parameters():
            p.requires_grad_(True)


class CompactCNN(PairedSequenceModel):
    """Compact dilated CNN baseline with the identical paired scoring API."""
    def __init__(self, config=None):
        config = config or ModelConfig(mode='cnn')
        if config.mode != 'cnn':
            raise ValueError('CompactCNN requires mode=cnn')
        super().__init__(config)


def build_model(config):
    """Construct the configured routed or LegNet architecture."""
    if config.architecture == 'legnet':
        from .legnet import PairedLegNet
        return PairedLegNet(config)
    return PairedSequenceModel(config)
