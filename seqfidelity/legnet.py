"""LegNet architecture and shared paired sequence scoring.

Architecture and initialization from autosome-ru/human_legnet, revision
 a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d.
Source: https://github.com/autosome-ru/human_legnet/blob/a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d/model.py
Copyright 2024 Dmitry Penzar, MIT; see licenses/human_legnet-MIT.txt.

The source architecture uses shared allele scoring with a float32 final head
and subtraction. Unequal sequence lengths are grouped during inference.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
    
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(inp, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), inp),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):
    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor
        
        block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       SELayer(self.inner_dim, reduction=self.se_reduction),
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.in_ch),
                       activation(),
        )
        
        self.block = block
    
    def forward(self, x):
        return self.block(x)
    
class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        
        self.block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.out_ch,
                            kernel_size=self.ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.out_ch),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)
    
class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)

class MapperBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_channels=in_features,
                      out_channels=out_features, 
                      kernel_size=1),
        )
        
    def forward(self, x):
        return self.block(x) 

class LegNet(nn.Module):
    def __init__(self, 
                 in_ch,
                 stem_ch,
                 stem_ks, 
                 ef_ks,
                 ef_block_sizes,
                 pool_sizes,
                 resize_factor,
                 activation=nn.SiLU,
                 ):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)
        
        self.in_ch = in_ch
        self.stem = LocalBlock(in_ch=in_ch, 
                               out_ch=stem_ch,
                               ks=stem_ks,
                               activation=activation)
        
        blocks = []
       
        in_ch = stem_ch
        out_ch = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch, 
                        out_ch=in_ch,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation)
                ),
                LocalBlock(in_ch=in_ch * 2,
                           out_ch=out_ch,
                           ks=ef_ks,
                           activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)
        
        self.mapper = MapperBlock(in_features=out_ch, 
                                  out_features=out_ch * 2)
        self.head = nn.Sequential(nn.Linear(out_ch * 2, out_ch * 2),
                                   nn.BatchNorm1d(out_ch * 2),
                                   activation(),
                                   nn.Linear(out_ch * 2, 1))
            
    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x =  F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


LEGNET_SOURCE = {
    'repository': 'https://github.com/autosome-ru/human_legnet',
    'revision': 'a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d',
    'architecture': {'stem_ch': 64, 'stem_ks': 11, 'ef_ks': 9,
                     'ef_block_sizes': [80, 96, 112, 128],
                     'resize_factor': 4, 'pool_sizes': [2, 2, 2, 2]},
    'initialization': 'official initialize_weights applied once after module construction',
    'adaptation': 'native input length, shared paired scoring and float32 head',
    'input_channels': 'A,G,C,T (official order); N is all-zero as official training Seq2Tensor output',
    'mixed_lengths': 'group by true length at inference; no fabricated flank padding',
}


class PairedLegNet(nn.Module):
    """Author architecture with the shared paired-activity training API."""
    def __init__(self, config):
        super().__init__()
        if config.architecture != 'legnet':
            raise ValueError('PairedLegNet requires architecture=legnet')
        self.config = config
        self.net = LegNet(in_ch=4, **LEGNET_SOURCE['architecture'])
        self.net.apply(initialize_weights)
        self.provenance = LEGNET_SOURCE

    @staticmethod
    def one_hot(tokens):
        # SeqFidelity A,C,G,T,N -> author A,G,C,T,N channel ordering.
        mapping = torch.tensor([0, 2, 1, 3, 4], device=tokens.device)
        return F.one_hot(mapping[tokens], num_classes=5)[..., :4].float().transpose(1, 2)

    def _score_equal_length(self, tokens):
        x = self.one_hot(tokens)
        x = self.net.mapper(self.net.main(self.net.stem(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.net.head(x.float()).squeeze(-1)

    def forward(self, tokens, lengths):
        unique = torch.unique(lengths)
        if len(unique) == 1:
            return self._score_equal_length(tokens[:, :int(unique[0])])
        if self.training:
            raise ValueError('MPRA-LegNet training requires equal actual construct lengths per batch')
        values = torch.zeros(len(tokens), device=tokens.device, dtype=torch.float32)
        for length in unique.tolist():
            rows = torch.nonzero(lengths == length, as_tuple=True)[0]
            values[rows] = self._score_equal_length(tokens[rows, :length])
        return values

    def forward_pair(self, ref, alt, lengths, return_features=False, zero_residual=False):
        if zero_residual or return_features:
            raise ValueError('Routing interventions and representation probes require a routed model')
        predictions = self(torch.cat((ref, alt)), torch.cat((lengths, lengths)))
        y_ref, y_alt = predictions.split(len(ref))
        return {'y_ref': y_ref, 'y_alt': y_alt, 'delta': y_alt.float()-y_ref.float()}

    def parameter_count(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
