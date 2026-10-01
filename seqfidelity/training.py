"""Paired sequence training, deterministic data loading, and checkpoints."""
from __future__ import annotations
import json, math, random, time, os
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .encoding import PairDataset, collate_pairs, loader, device_batch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')



def json_clean(value):
    if isinstance(value, dict): return {str(k): json_clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray): return json_clean(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value



def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(value), indent=2, ensure_ascii=False)+'\n')



@dataclass
class TrainConfig:
    seed:int=0
    batch_size:int=64
    epochs:int=16
    min_epochs:int=5
    patience:int=4
    learning_rate:float=3e-4
    weight_decay:float=1e-2
    absolute_weight:float=.1
    objective:str='delta_activity'
    reverse_complement:bool=False
    amp:bool=True
    device:str='cpu'
    clip_grad_norm:float=1.
    compile_training:bool=False
    gradient_diagnostic_epochs:tuple[int,...]=()



def macro_mse(frame,predictions):
    if not np.isfinite(np.asarray(predictions)).all():
        raise FloatingPointError('Non-finite predictions cannot be omitted from validation')
    errors=pd.DataFrame({'locus_id':frame.locus_id.to_numpy(),
                         'error':(frame.delta.to_numpy()-np.asarray(predictions))**2})
    return float(errors.groupby('locus_id',sort=False).error.mean().mean())



@torch.inference_mode()
def predict(model,frame,cfg,zero_residual=False):
    model.eval(); device=torch.device(cfg.device)
    values={'delta_pred':[],'y_ref_pred':[],'y_alt_pred':[],'k_ref':[],'k_alt':[]}
    for raw in loader(frame,cfg.batch_size):
        batch=device_batch(raw,device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=cfg.amp and device.type=='cuda'):
            out=model.forward_pair(batch['ref'],batch['alt'],batch['lengths'],zero_residual=zero_residual)
        for key,source in [('delta_pred','delta'),('y_ref_pred','y_ref'),('y_alt_pred','y_alt')]:
            values[key].extend(out[source].float().cpu().numpy().tolist())
        for key,source in [('k_ref','route_ref'),('k_alt','route_alt')]:
            route=out.get(source)
            if route is not None:
                count=route.selected_lengths
                values[key].extend(count.cpu().tolist())
    result=frame.drop(columns=['ref_seq','alt_seq'],errors='ignore').copy()
    if 'delta' in result:
        result['delta_true']=result.delta
    for key,value in values.items():
        if value: result[key]=value
    return result



def loss_for(out,batch,cfg,delta_scale,activity_scale):
    if cfg.objective not in {'delta_activity', 'activity'}:
        raise ValueError('objective must be delta_activity or activity')
    if delta_scale <= 0 or activity_scale <= 0:
        raise ValueError('Loss scales must be positive')
    dloss=((out['delta'].float()-batch['delta'])/delta_scale).square().mean()
    activity=batch['activity']; valid=torch.isfinite(activity)
    predictions=torch.stack((out['y_ref'].float(),out['y_alt'].float()),-1)
    aloss=((predictions[valid]-activity[valid])/activity_scale).square().mean() if valid.any() else predictions.sum()*0.
    if cfg.objective=='activity':
        if not valid.any(): raise ValueError('Activity-only training requires measured activity labels')
        return aloss,dloss,aloss
    return dloss+cfg.absolute_weight*aloss,dloss,aloss



def branch_fusion_snapshot(model, batch, cfg, delta_scale, activity_scale, stage):
    """Fixed-batch gradient and RMS branch-fusion diagnostics.

    No optimizer update; restore execution RNG and model mode. Parameter
    gradients use the existing unclipped standardized-loss diagnostic.
    """
    from .interventions import gradient_snapshot
    from .model import length_mask
    device = batch['ref'].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        with torch.random.fork_rng(devices=devices):
            result = gradient_snapshot(model, batch, cfg, delta_scale, activity_scale, stage)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                    enabled=cfg.amp and device.type == 'cuda'):
                out = model.forward_pair(batch['ref'], batch['alt'], batch['lengths'], return_features=True)
                fusion = {}
                for allele in ('ref', 'alt'):
                    features = out['features_'+allele]
                    u, bypass = features['dechunked'], features['bypass']
                    valid = length_mask(batch['lengths'], u.shape[1])
                    uf, sf = u.float()[valid], bypass.float()[valid]
                    fused = model.fuse_branches(u, bypass).float()[valid]
                    denominator = uf.norm(dim=-1) * sf.norm(dim=-1)
                    usable = denominator > 0
                    cosine = (uf*sf).sum(-1)[usable] / denominator[usable]
                    fusion[allele] = {
                        'u_rms': float(uf.square().mean().sqrt()),
                        'projected_encoder_rms': float(sf.square().mean().sqrt()),
                        'fused_rms': float(fused.square().mean().sqrt()),
                        'mean_position_cosine_u_s': float(cosine.mean()) if usable.any() else None,
                        'cosine_defined_positions': int(usable.sum()),
                        'valid_positions': int(valid.sum()),
                    }
                result.update(branch_fusion=model.config.branch_fusion,
                    external_skip=model.config.external_skip, fusion=fusion,
                    batch_selection='First batch of original ordered training dataframe; no shuffle or augmentation.',
                    diagnostic_mode='eval, with autograd for unclipped parameter gradients; no optimizer update')
        return result
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)



def train_model(model_config,cfg,train_frame,validation_frame,output_dir,checkpoint_root=None):
    from .model import build_model, ModelConfig
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    for name, frame in [('train', train_frame), ('validation', validation_frame)]:
        if frame.empty or not {'delta', 'locus_id'}.issubset(frame.columns):
            raise ValueError(f'{name} requires labeled rows with locus_id and delta')
        if not np.isfinite(frame.delta.to_numpy(float)).all() or frame.locus_id.isna().any():
            raise ValueError(f'{name} labels must be finite and locus identifiers must be present')
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=(output_dir/'checkpoints' if checkpoint_root is None else Path(checkpoint_root)/output_dir.name)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    best_path,last_path=checkpoint_dir/'best.pt',checkpoint_dir/'last.pt'
    seed_everything(cfg.seed)
    model=build_model(model_config).to(cfg.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    delta_scale=max(float(np.sqrt(np.mean(train_frame.delta.to_numpy()**2))),.01)
    activity=train_frame.reindex(columns=['y_ref','y_alt']).to_numpy(dtype=float)
    activity_scale=max(float(np.nanstd(activity)),.1) if np.isfinite(activity).any() else 1.
    metadata={'model':asdict(model_config),'training':asdict(cfg),'train_pairs':len(train_frame),
              'validation_pairs':len(validation_frame),'train_loci':train_frame.locus_id.nunique(),
              'validation_loci':validation_frame.locus_id.nunique(),'delta_scale':delta_scale,
              'activity_scale':activity_scale,'parameters':sum(p.numel() for p in model.parameters()),
              'torch':torch.__version__,'device':torch.cuda.get_device_name() if cfg.device=='cuda' else cfg.device,
              'checkpoint':str(best_path),'initialization_seed':cfg.seed,
              'selection':'minimum validation locus-macro delta MSE'}
    if hasattr(model,'provenance'):
        metadata['architecture_provenance']=model.provenance
    config_path=output_dir/'config.json'
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        previous['model']=asdict(ModelConfig(**previous['model']))
        previous['training']=asdict(TrainConfig(**previous['training']))
        if json_clean(previous)!=json_clean(metadata):
            raise ValueError(f'Configuration changed for existing run: {output_dir}')
    write_json(config_path,metadata)
    history,best_metric,bad_epochs,start_epoch=[],math.inf,0,0
    if last_path.exists():
        saved=torch.load(last_path,map_location=cfg.device,weights_only=False)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        history=saved['history']; best_metric,bad_epochs=saved['best_metric'],saved['bad_epochs']
        start_epoch=saved['epoch']+1; torch.set_rng_state(saved['rng_cpu'].cpu())
        if cfg.device=='cuda': torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
    dataset=PairDataset(train_frame); start=time.perf_counter()
    fusion_diagnostic_batch = None
    if model_config.branch_fusion == 'rms_balanced':
        fixed_raw = collate_pairs([dataset[i] for i in range(min(cfg.batch_size, len(dataset)))])
        fusion_diagnostic_batch = device_batch(fixed_raw, torch.device(cfg.device))
    def record_fusion_diagnostic(stage, selected_epoch=None):
        if fusion_diagnostic_batch is None:
            return
        diagnostic = branch_fusion_snapshot(model, fusion_diagnostic_batch, cfg,
                                            delta_scale, activity_scale, stage)
        if selected_epoch is not None:
            diagnostic['selected_epoch'] = selected_epoch
        path = output_dir/'branch_fusion_diagnostics.json'
        snapshots = json.loads(path.read_text()) if path.exists() else []
        snapshots = [row for row in snapshots if row['stage'] != stage]
        snapshots.append(diagnostic)
        write_json(path, snapshots)
    if start_epoch == 0:
        record_fusion_diagnostic('initialization', selected_epoch=0)
    train_forward=model.forward_pair
    if cfg.compile_training and cfg.device=='cuda':
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
        train_forward=torch.compile(model.forward_pair,mode='reduce-overhead',fullgraph=True)
    for epoch in range(start_epoch,cfg.epochs):
        if bad_epochs>=cfg.patience and epoch>=cfg.min_epochs: break
        model.train(); totals=np.zeros(3); seen=0; epoch_start=time.perf_counter()
        for step,raw in enumerate(loader(dataset,cfg.batch_size,shuffle=True,seed=cfg.seed+epoch)):
            batch=device_batch(raw,torch.device(cfg.device),cfg.reverse_complement)
            if step==0 and epoch+1 in cfg.gradient_diagnostic_epochs:
                from .interventions import gradient_snapshot
                diagnostic=gradient_snapshot(model,batch,cfg,delta_scale,activity_scale,
                                             stage=f'epoch_{epoch+1}_first_batch')
                diagnostic_path=output_dir/'gradient_diagnostics.json'
                snapshots=json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else []
                snapshots=[row for row in snapshots if row.get('stage')!=diagnostic['stage']]
                snapshots.append(diagnostic)
                write_json(diagnostic_path,snapshots)
            if cfg.compile_training and cfg.device=='cuda':
                torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(cfg.device).type,dtype=torch.bfloat16,enabled=cfg.amp and cfg.device=='cuda'):
                out=train_forward(batch['ref'],batch['alt'],batch['lengths'])
                loss,dloss,aloss=loss_for(out,batch,cfg,delta_scale,activity_scale)
            if not torch.isfinite(loss): raise FloatingPointError(f'Non-finite training loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_grad_norm,error_if_nonfinite=True)
            optimizer.step(); count=len(batch['lengths'])
            totals+=np.array([loss.item(),dloss.item(),aloss.item()])*count; seen+=count
            if (step+1)%250==0:
                print(json.dumps({'run':output_dir.name,'epoch':epoch+1,'pairs_seen':seen,'loss':totals[0]/seen,'seconds':time.perf_counter()-epoch_start}),flush=True)
        predictions=predict(model,validation_frame,cfg)
        metric=macro_mse(validation_frame,predictions.delta_pred)
        entry={'epoch':epoch+1,'train_loss':totals[0]/seen,'train_delta_normalized_mse':totals[1]/seen,
               'train_activity_normalized_mse':totals[2]/seen,'validation_macro_delta_mse':metric,
               'seconds':time.perf_counter()-epoch_start,'pairs_seen':seen}
        history.append(entry)
        if metric<best_metric:
            best_metric,bad_epochs=metric,0
            torch.save({'state_dict':model.state_dict(),'model_config':asdict(model_config),
                        'train_config':asdict(cfg),'epoch':epoch+1,'validation_macro_delta_mse':metric},best_path)
            predictions.to_csv(output_dir/'validation_predictions.tsv',sep='\t',index=False)
        else: bad_epochs+=1
        if epoch == 0:
            record_fusion_diagnostic('epoch_1_end', selected_epoch=1)
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'history':history,'best_metric':best_metric,'bad_epochs':bad_epochs,
                    'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state() if cfg.device=='cuda' else None},last_path)
        write_json(output_dir/'history.json',history)
        print(json.dumps({'run':output_dir.name,**entry}),flush=True)
    saved=torch.load(best_path,map_location=cfg.device,weights_only=False)
    model.load_state_dict(saved['state_dict']); model.eval()
    record_fusion_diagnostic('selected_best', selected_epoch=saved['epoch'])
    write_json(output_dir/'training_summary.json',{
        'epochs_completed':len(history),'best_epoch':saved['epoch'],'best_validation_macro_delta_mse':best_metric,
        'training_seconds':sum(row['seconds'] for row in history),'this_invocation_seconds':time.perf_counter()-start,
        'early_stopped':len(history)<cfg.epochs,'max_epochs_reached':len(history)==cfg.epochs,
        'zero_predictor_validation_macro_delta_mse':macro_mse(validation_frame,np.zeros(len(validation_frame)))})
    return model



def load_model(checkpoint,device='cpu'):
    from .model import ModelConfig,build_model
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    model=build_model(ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['state_dict']); model.eval()
    cfg=TrainConfig(**saved['train_config']);cfg.device=device
    return model,cfg

