"""Train token-level shared readouts on detached compressed representations."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from .model import length_mask
from .precision import execution_precision
from .encoding import collate_pairs, device_batch
from .training import write_json


@dataclass
class TokenProbeConfig:
    seed:int=0
    epochs:int=20
    min_epochs:int=5
    patience:int=4
    batch_size:int=4096
    micro_batch_size:int=128
    learning_rate:float=.001
    weight_decay:float=.0001
    clip_grad_norm:float=1.
    device:str='cpu'
    n_bootstrap:int=1000



def sync_device(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)



@torch.no_grad()
def extract_compressed_prefix(model,ref,alt,lengths,*,bf16=True):
    """Ordinary detached tensors permit head backward; no backbone is evaluated."""
    if model.config.mode not in ('none','fixed','learned'):
        raise ValueError('Only none/fixed/learned compressed prefixes are supported')
    model.eval();both_lengths=torch.cat((lengths,lengths))
    with execution_precision(ref.device.type,bf16):
        encoded=model.encode(torch.cat((ref,alt)),both_lengths)
        route=model.make_route(encoded,both_lengths)
        selected=encoded.gather(1,route.indices[...,None].expand(-1,-1,encoded.shape[-1]))
        selected=selected*route.selected_mask[...,None]
        if model.config.mode=='none':selected=encoded
    return {'hidden':selected.detach(),'coords':route.coords.detach(),
            'lengths':route.selected_lengths.detach(),'original_lengths':both_lengths,'pair_count':len(ref)}



def compressed_input_identity(prefix):
    """Read-only equality before normalization, in the current extraction precision."""
    b=prefix['pair_count'];h=prefix['hidden'];coords=prefix['coords'];lengths=prefix['lengths']
    mask=length_mask(lengths,h.shape[1]);r,a=mask[:b],mask[b:]
    same_mask=(r==a).all(1)&lengths[:b].eq(lengths[b:])
    common=r&a
    values=same_mask&((h[:b]==h[b:])|~common[...,None]).flatten(1).all(1)
    positions=same_mask&((coords[:b]==coords[b:])|~common).all(1)
    same_native=prefix['original_lengths'][:b].eq(prefix['original_lengths'][b:])
    return {'compressed_values_identical':values,'compressed_coords_identical':positions,
            'compressed_input_identical':values&positions&same_native}



def token_pool(hidden,coords,lengths,original_lengths):
    mask=length_mask(lengths,hidden.shape[1])[...,None]
    h=hidden.float().masked_fill(~mask,0);count=lengths[:,None].float()
    u=coords.float()/(original_lengths[:,None]-1).clamp_min(1);angle=2*torch.pi*u[...,None]
    return torch.cat((h.sum(1)/count,h.masked_fill(~mask,-torch.inf).amax(1),
                      (h*angle.sin()).sum(1)/count,(h*angle.cos()).sum(1)/count),dim=-1)



class TokenDeltaProbe(nn.Module):
    """Convolutional shared-allele readout followed by alternate minus reference."""
    def __init__(self,mean,std,conv_channels=64,hidden=192):
        super().__init__()
        channels = torch.as_tensor(mean).numel()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).reshape(channels))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).reshape(channels))
        self.conv=nn.Conv1d(channels,conv_channels,3,padding=1,bias=True)
        self.gelu=nn.GELU();self.fc=nn.Linear(4*conv_channels,hidden,bias=False);self.out=nn.Linear(hidden,1,bias=True)
    def allele_scores(self,hidden,coords,lengths,original_lengths):
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            mask=length_mask(lengths,hidden.shape[1])[...,None]
            x=((hidden.float()-self.mean)/self.std).masked_fill(~mask,0)
            x=self.gelu(self.conv(x.transpose(1,2)).transpose(1,2)).masked_fill(~mask,0)
            pooled=token_pool(x,coords,lengths,original_lengths)
            return self.out(self.gelu(self.fc(pooled))).squeeze(-1)
    def forward(self,prefix):
        score=self.allele_scores(prefix['hidden'],prefix['coords'],prefix['lengths'],prefix['original_lengths'])
        ref,alt=score.split(prefix['pair_count'])
        return alt-ref



def microbatches(dataset,order,batch_size):
    for start in range(0,len(order),batch_size):
        yield collate_pairs([dataset[int(i)] for i in order[start:start+batch_size]])



def timed_prefix(model,raw,device,*,bf16):
    batch=device_batch(raw,torch.device(device));sync_device(device);start=perf_counter()
    prefix=extract_compressed_prefix(model,batch['ref'],batch['alt'],batch['lengths'],bf16=bf16)
    sync_device(device)
    return prefix,batch,perf_counter()-start



def update_moments(count,mean,m2,values):
    variance,batch_mean=torch.var_mean(values.float(),dim=0,unbiased=False)
    n=len(values);bmean=batch_mean.double().cpu().numpy();bm2=variance.double().cpu().numpy()*n
    difference=bmean-mean;total=count+n
    return total,mean+difference*(n/total),m2+bm2+difference*difference*(count*n/total)



def channel_statistics(model,dataset,checkpoint,path,*,device='cpu',micro_batch_size=128,bf16=True):
    checkpoint,path=Path(checkpoint).resolve(),Path(path);encoder=checkpoint.parent.name
    identity={'encoder':encoder,'encoder_checkpoint':str(checkpoint),
        'encoder_checkpoint_mtime_ns':checkpoint.stat().st_mtime_ns,'train_rows':len(dataset),
        'micro_batch_size':micro_batch_size,'bf16':bf16,'mode':model.config.mode,'compression':model.config.compression,
        'normalization':'Per-channel population mean/std over valid train ref+alt compressed tokens only.',
        'variance_rule':'variance<=1e-12: scale1; otherwise max(std,1e-4), matching pooled numeric floor.',
        'identity_scope':'Before normalization, actual precision, valid masks/lengths and raw retained coordinates.'}
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous.get(k)!=v for k,v in identity.items()):raise ValueError('Channel statistics identity changed')
        if previous.get('status')=='complete':return previous
    sync_device(device);started=perf_counter();count=0;mean=np.zeros(model.feature_dim,np.float64);m2=np.zeros(model.feature_dim,np.float64)
    prefix_seconds=0.;identical={name:0 for name in ('compressed_values_identical','compressed_coords_identical','compressed_input_identical')}
    for step,raw in enumerate(microbatches(dataset,np.arange(len(dataset)),micro_batch_size)):
        prefix,_,elapsed=timed_prefix(model,raw,device,bf16=bf16);prefix_seconds+=elapsed
        valid=length_mask(prefix['lengths'],prefix['hidden'].shape[1])
        count,mean,m2=update_moments(count,mean,m2,prefix['hidden'][valid])
        for name,flag in compressed_input_identity(prefix).items():identical[name]+=int(flag.sum().cpu())
        if (step+1)%250==0:print(f'token stats {encoder}: {min((step+1)*micro_batch_size,len(dataset))}/{len(dataset)}',flush=True)
    variance=np.maximum(m2/count,0);std=np.where(variance<=1e-12,1.,np.maximum(np.sqrt(variance),1e-4))
    y=dataset.delta
    if not np.isfinite(y).all():raise ValueError('All supplied delta labels must be finite')
    target_scale=max(float(np.sqrt(np.mean(y**2))),.01);sync_device(device)
    result={**identity,'status':'complete','valid_tokens_total':count,'alleles':len(dataset)*2,
        'mean':mean.astype(np.float32).tolist(),'std':std.astype(np.float32).tolist(),'target_scale':target_scale,
        'batch_moments_dtype':'FP32 batch mean/variance, FP64 weighted merges',
        'training_identity_counts':{'n_pairs':len(dataset),**identical},
        'timings':{'total_seconds':perf_counter()-started,'prefix_seconds':prefix_seconds},
        'timing_scope':'Shared train statistics once per encoder; prefix is a subset of total.',
        'raw_token_cache':False,'dataset_order':'supplied training order, ref then alt within each paired microbatch'}
    write_json(path,result);return result



def train_token_epoch(model,probe,dataset,optimizer,order,target_scale,config,*,bf16=True):
    """Sum/n_update accumulation, including the final actual-size update."""
    model.eval();probe.train();sync_device(config.device);started=perf_counter()
    total_loss,prefix_seconds,seen=0.,0.,0;update_sizes=[]
    for start in range(0,len(order),config.batch_size):
        update_order=order[start:start+config.batch_size];n_update=len(update_order)
        optimizer.zero_grad(set_to_none=True)
        for raw in microbatches(dataset,update_order,config.micro_batch_size):
            prefix,batch,elapsed=timed_prefix(model,raw,config.device,bf16=bf16);prefix_seconds+=elapsed
            prediction=probe(prefix);loss_sum=(prediction-batch['delta'].float()/target_scale).square().sum()
            if not torch.isfinite(loss_sum):raise FloatingPointError('Nonfinite token-probe loss')
            (loss_sum/n_update).backward()
            total_loss+=float(loss_sum.detach().cpu());seen+=len(raw['index'])
        torch.nn.utils.clip_grad_norm_(probe.parameters(),config.clip_grad_norm,error_if_nonfinite=True)
        optimizer.step();update_sizes.append(n_update)
    sync_device(config.device)
    return {'train_normalized_mse':total_loss/seen,'training_seconds':perf_counter()-started,
        'training_prefix_seconds':prefix_seconds,'pairs_seen':seen,'optimizer_updates':len(update_sizes),
        'optimizer_update_pair_counts':update_sizes}



@torch.no_grad()
def token_predictions(model,probe,dataset,target_scale,config,*,bf16=True):
    model.eval();probe.eval();sync_device(config.device);started=perf_counter()
    values=[];prefix_seconds=0.;flags={name:[] for name in ('compressed_values_identical','compressed_coords_identical','compressed_input_identical')}
    for raw in microbatches(dataset,np.arange(len(dataset)),config.micro_batch_size):
        prefix,_,elapsed=timed_prefix(model,raw,config.device,bf16=bf16);prefix_seconds+=elapsed
        values.append((probe(prefix)*target_scale).float().cpu().numpy())
        for name,flag in compressed_input_identity(prefix).items():flags[name].append(flag.cpu().numpy())
    prediction=np.concatenate(values)
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite token prediction; no rows dropped')
    sync_device(config.device)
    return prediction,{'wall_seconds':perf_counter()-started,'prefix_seconds':prefix_seconds},\
        {name:np.concatenate(parts) for name,parts in flags.items()}

