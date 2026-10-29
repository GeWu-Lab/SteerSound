"""Standard flow-matching pretraining and fine-tuning with optional validation."""
from __future__ import annotations
import argparse,gc,json,math,os,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from safetensors.torch import load_file,save_file
from steersound import SteerSoundPipeline
from steersound.models import Edit_Flux
from steersound.configuration import dit_params
from dataset.worker_determinism import EpochSeededDataset,formal_worker_init_fn,dataloader_generator
from .task_aware_sigma import TaskAwareSigmaSampler
from .kstep_objective import HybridTaskSigmaSampler
from .stage_patience import RecoveryPatience
from .objective import prepare,loss
from .evaluate import evaluate,selection_score


def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def load_dit(model,path):
    path=Path(path)
    if path.is_dir():path=path/'dit/model.safetensors'
    if path.suffix=='.safetensors':state=load_file(str(path))
    else:
        state=torch.load(path,map_location='cpu',weights_only=True)
        state=state.get('module',state.get('model',state))
        if any(k.startswith('dit.') for k in state):state={k[4:]:v for k,v in state.items() if k.startswith('dit.')}
    model.load_state_dict(state,strict=True)

def save_dit(model,path,rank):
    if rank==0:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path))
    dist.barrier()

def initialize_editor(pipe,mode,checkpoint,seed):
    if mode=='pretrain':
        del pipe.dit;gc.collect();torch.cuda.empty_cache()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);pipe.dit=Edit_Flux(dit_params).to(device=pipe.device,dtype=pipe.dtype)
    else:
        load_dit(pipe.dit,checkpoint)

def make_sigma_sampler(mode,seed,rank):
    if mode=='recovery':return HybridTaskSigmaSampler(seed=seed,rank=rank)
    return TaskAwareSigmaSampler('logit' if mode in ('pretrain','finetune') else 'routed',seed=seed,rank=rank)

def parser(mode,setting):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='GeWu-Lab/SteerSound',help='Frozen VAE, T5, vocoder and architecture assets')
    p.add_argument('--dit',type=Path,help='Editor checkpoint for fine-tuning; omitted when pretraining a new DiT')
    p.add_argument('--online-config',required=True,type=Path,help='Prepared source-audio config for online editing-example construction')
    p.add_argument('--val',type=Path,help='Optional paired Complex validation folder for standard fine-tuning')
    p.add_argument('--test',type=Path,help='Optional paired Complex test folder; requires --val and --baseline-test')
    p.add_argument('--baseline-val',type=Path,help='Initialization-checkpoint validation summary containing primary six-task scores')
    p.add_argument('--baseline-test',type=Path,help='Initialization-checkpoint test summary; never used for selection')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=70 if mode=='pretrain' else 3 if mode=='finetune' else (20 if mode=='emphasis' else 2 if setting=='complex' else 5))
    p.add_argument('--batch-size',type=int,default=1 if mode in ('pretrain','finetune') else 30,help='Per GPU; global batch = batch * GPUs * accumulation')
    p.add_argument('--accumulation',type=int,default=1);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4 if mode=='pretrain' else 2e-6 if mode in ('emphasis','finetune') else 1e-5)
    p.add_argument('--warmup',type=int,default=64 if mode=='recovery' else 1000)
    p.add_argument('--seed',type=int,default=20260829);p.add_argument('--eval-seed',type=int,default=20260804)
    p.add_argument('--eval-every',type=float,default=.5 if setting=='complex' else 1.,help='Epoch interval: 0.5 or 1')
    p.add_argument('--max-k',type=int,default=15);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--save-every',type=int,default=5);p.add_argument('--metric-device',default='cpu')
    p.add_argument('--precision',choices=['bfloat16','float16'],default='bfloat16')
    p.add_argument('--local_rank','--local-rank',type=int,default=-1)
    return p

def parse_args(mode,setting,argv=None):
    p=parser(mode,setting);a=p.parse_args(argv)
    if min(a.epochs,a.batch_size,a.accumulation,a.save_every,a.patience)<1 or a.workers<0 or not 1<=a.max_k<=15:p.error('Invalid positive training parameter')
    if a.warmup<2:p.error('warmup must be at least 2 optimizer updates')
    if a.eval_every not in (.5,1.):p.error('eval-every must be 0.5 or 1')
    if mode=='finetune':
        if a.dit is None:p.error('--dit is required to initialize standard fine-tuning')
        if a.test is not None and a.val is None:p.error('--test requires --val for checkpoint selection')
        if setting=='complex' and a.val is not None and a.baseline_val is None:p.error('Complex validation needs --baseline-val from the initialization checkpoint')
        if setting=='complex' and a.test is not None and a.baseline_test is None:p.error('Complex test needs its own --baseline-test')
    elif mode!='pretrain':
        if a.dit is None or a.val is None:p.error('--dit and --val are required for fine-tuning')
        if setting=='complex' and a.baseline_val is None:p.error('Complex validation needs --baseline-val from the initialization checkpoint')
        if setting=='complex' and a.test is not None and a.baseline_test is None:p.error('Complex test needs its own --baseline-test')
    if mode=='pretrain' and any(x is not None for x in (a.val,a.test,a.dit)):p.error('Pretraining starts a new DiT and does not use val/test')
    return a

def main(mode,setting):
    a=parse_args(mode,setting)
    use_validation=mode!='pretrain' and a.val is not None
    if not torch.cuda.is_available():raise RuntimeError('Training requires a CUDA-enabled PyTorch build and NVIDIA GPUs')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    import deepspeed
    deepspeed.init_distributed(dist_backend='nccl')
    rank,world=dist.get_rank(),dist.get_world_size();device=f'cuda:{local}'
    random.seed(a.seed+rank);np.random.seed((a.seed+rank)%2**32);torch.manual_seed(a.seed+rank)
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    pipe=SteerSoundPipeline.from_pretrained(a.model,device=device,dtype=getattr(torch,a.precision))
    initialize_editor(pipe,mode,a.dit,a.seed)
    from dataset.dataset import AudioDataset,collate_fn
    base=AudioDataset(**json.loads(a.online_config.read_text()))
    data=EpochSeededDataset(base,training_seed=a.seed,rank=rank)
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True)
    loader=DataLoader(data,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,drop_last=True,
        collate_fn=collate_fn,pin_memory=True,persistent_workers=a.workers>0,
        multiprocessing_context='spawn' if a.workers else None,
        worker_init_fn=formal_worker_init_fn if a.workers else None,generator=dataloader_generator(a.seed,rank))
    updates=len(loader)//a.accumulation
    if updates<1:raise ValueError('Not enough samples for one global optimizer update')
    def read_baseline(path,split):
        if path is None:return None
        summary=json.loads(path.read_text())
        if summary.get('setting')!='complex' or summary.get('split')!=split:raise ValueError(f'Baseline must be Complex {split}')
        return summary['primary']
    baseline_val=read_baseline(a.baseline_val,'val')
    baseline_test=read_baseline(a.baseline_test,'test')
    if rank==0:
        atomic_json(a.output/'training_config.json',dict(mode=mode,setting=setting,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            world_size=world,global_batch=a.batch_size*world*a.accumulation,updates_per_epoch=updates,selection_split='val' if use_validation else None))
    best_path=a.output/'selected.safetensors';best_score=None
    if use_validation:
        report=evaluate(pipe,a.val,setting,a.output/'val_initial',split='val',baseline=baseline_val,seed=a.eval_seed,metric_device=a.metric_device)
        best_score=selection_score(report);save_dit(pipe.dit,best_path,rank)
    patience=RecoveryPatience(best_score=best_score,patience=a.patience) if mode=='recovery' else None
    stages=range(1,a.max_k+1) if mode=='recovery' else [0]
    history=[];selected=dict(k=0,epoch=0,score=best_score)
    for k in stages:
        if mode=='recovery':
            del pipe.dit;gc.collect();torch.cuda.empty_cache()
            pipe.dit=Edit_Flux(dit_params)
            load_dit(pipe.dit,best_path)
            pipe.dit=pipe.dit.to(device=device,dtype=pipe.dtype)
        pipe.dit.requires_grad_(True).train()
        # A new optimizer and schedule at EVERY recovery depth, including k=1.
        optimizer=torch.optim.AdamW(pipe.dit.parameters(),lr=a.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        cfg=dict(train_micro_batch_size_per_gpu=a.batch_size,train_batch_size=a.batch_size*world*a.accumulation,
            gradient_accumulation_steps=a.accumulation,gradient_clipping=1.,zero_optimization=dict(stage=2,overlap_comm=True,contiguous_gradients=True),
            bf16=dict(enabled=a.precision=='bfloat16'),fp16=dict(enabled=a.precision=='float16'),steps_per_print=100)
        cfg['scheduler']=(dict(type='WarmupLR',params=dict(warmup_min_lr=0.,warmup_max_lr=a.lr,warmup_num_steps=a.warmup,warmup_type='linear'))
            if mode=='recovery' else dict(type='WarmupCosineLR',params=dict(total_num_steps=updates*a.epochs,warmup_num_steps=a.warmup,cos_min_ratio=1e-5)))
        engine,optimizer,_,scheduler=deepspeed.initialize(model=pipe.dit,model_parameters=pipe.dit.parameters(),optimizer=optimizer,config=cfg)
        if mode=='recovery':scheduler.step(last_batch_iteration=1)
        pipe.dit=engine.module
        sigma=make_sigma_sampler(mode,a.seed,rank)
        noise=torch.Generator(device=device).manual_seed(a.seed+rank*1_000_003+(k*10_000_019 if mode=='recovery' else 19_991))
        for epoch in range(1,a.epochs+1):
            dataset_epoch=epoch-1
            data.set_epoch(dataset_epoch);sampler.set_epoch(dataset_epoch)
            usable=updates*a.accumulation
            boundaries={usable} if a.eval_every==1 else {max(1,math.ceil(updates/2))*a.accumulation,usable}
            for micro,batch in enumerate(loader,1):
                if micro>usable:break
                prepared=prepare(batch,pipe,sigma,noise)
                objective=loss(engine,prepared,k if mode=='recovery' else None)
                if not bool(torch.isfinite(objective)):raise RuntimeError(f'Nonfinite loss k={k}, epoch={epoch}, micro={micro}')
                engine.backward(objective);engine.step()
                if micro in boundaries and use_validation:
                    fraction=epoch-1+(0.5 if micro<usable else 1.0);label=f'k{k:02d}_e{fraction:.6f}'
                    report=evaluate(pipe,a.val,setting,a.output/label,split='val',baseline=baseline_val,seed=a.eval_seed,metric_device=a.metric_device)
                    score=selection_score(report);improved=score>best_score
                    if improved:
                        best_score=score;selected=dict(k=k,epoch=fraction,score=score)
                        save_dit(pipe.dit,best_path,rank)
                    history.append(dict(k=k,epoch=fraction,score=score,improved=improved))
                    if rank==0:
                        atomic_json(a.output/'history.json',history);atomic_json(a.output/'selection.json',dict(**selected,split='val',test_used_for_selection=False,tie_break='retain input or earliest checkpoint'))
                    engine.train()
            checkpoint=a.output/f'checkpoint_k{k:02d}_epoch{epoch:03d}'
            if mode!='pretrain' or epoch%a.save_every==0 or epoch==a.epochs:
                engine.save_checkpoint(str(checkpoint),tag='training',client_state=dict(epoch=epoch,k=k,sigma=sigma.state_dict(),noise_generator=noise.get_state()))
                if mode in ('pretrain','finetune'):save_dit(pipe.dit,a.output/f'dit_epoch{epoch:03d}.safetensors',rank)
        # Release optimizer state before the next depth. ZeRO-2 has replicated parameters.
        del engine,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        if mode=='recovery':
            patience.update(k,best_score)
            stopped=patience.should_stop
            if rank==0:atomic_json(a.output/'patience.json',patience.state_dict())
            if stopped:break
    if use_validation:
        load_dit(pipe.dit,best_path);pipe.dit.requires_grad_(False).eval()
        if a.test:
            evaluate(pipe,a.test,setting,a.output/'test_final_once',split='test',baseline=baseline_test,seed=a.eval_seed,metric_device=a.metric_device)
        if rank==0:atomic_json(a.output/'selection.json',dict(**selected,split='val',test_used_for_selection=False,tie_break='retain input or earliest checkpoint'))
    elif mode=='finetune':
        save_dit(pipe.dit,a.output/'final.safetensors',rank)
    dist.barrier();dist.destroy_process_group()
