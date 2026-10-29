"""Original standard FM and detached k-step corrective FM for the base editor."""
from __future__ import annotations
from types import SimpleNamespace
import torch
from einops import rearrange,repeat
from steersound.codec import vae_encode
from .kstep_objective import SIGMAS,corrective_velocity_target,detached_rollout_to_k
from .fm_objective import standard_fm_mse

def pack(latent):
    padding=(2-latent.shape[2]%2)%2
    latent=torch.nn.functional.pad(latent,(0,0,0,padding))
    return rearrange(latent,'b c (h ph) (w pw) -> b (h w) (c ph pw)',ph=2,pw=2)

def unpack(tokens,reference):
    h,w=reference.shape[2:]
    return rearrange(tokens,'b (h w) (c ph pw) -> b c (h ph) (w pw)',h=(h+1)//2,ph=2,pw=2)[:,:,:h,:w]

def prepare(batch,pipe,sampler,noise_generator):
    with torch.no_grad():
        encoded=vae_encode(dict(batch),pipe.vae,SimpleNamespace(duration=10.),pipe.device,pipe.dtype)
        text,mask=pipe.encode_text(encoded['prompt_text'])
    # Inference-mode frozen encoders can return inference tensors; clone before autograd.
    text,mask=text.clone(),mask.clone()
    target=encoded['tar_audio'];reference=encoded['ref_audio'].to(pipe.dtype)
    tasks=encoded['task'].to(pipe.device).reshape(-1).long()
    sigma=sampler.sample(tasks,device=pipe.device,dtype=torch.float32).sigma
    noise=torch.randn(target.shape,generator=noise_generator,device=pipe.device,dtype=target.dtype)
    noisy=(1-sigma[:,None,None,None])*target+sigma[:,None,None,None]*noise
    b,_,h,w=target.shape
    ids=torch.zeros((h+1)//2,w//2,3,device=pipe.device)
    ids[...,1]+=torch.arange((h+1)//2,device=pipe.device)[:,None]
    ids[...,2]+=torch.arange(w//2,device=pipe.device)[None,:]
    cond=dict(ref_img=pack(reference),img_ids=repeat(ids,'h w c -> b (h w) c',b=b),txt=text,
        txt_ids=torch.zeros(b,text.shape[1],3,device=pipe.device),txt_mask=mask,
        task=tasks.to(pipe.dtype)/5,
        change_vol=(encoded['change_vol'].to(pipe.device,pipe.dtype).reshape(-1)+5)/10,
        change_Pitch=(encoded['change_Pitch'].to(pipe.device,pipe.dtype).reshape(-1)+4)/8)
    return dict(target=target,noise=noise,noisy=noisy.to(pipe.dtype),sigma=sigma.to(pipe.dtype),tasks=tasks,cond=cond)

def forward(model,prepared,latent,sigma):
    return model(noi_img=pack(latent),timesteps=sigma,**prepared['cond'])

def subset(prepared,mask):
    return {key:{k:v[mask] for k,v in value.items()} if key=='cond' else value[mask] for key,value in prepared.items()}

def loss(model,prepared,k=None):
    if k is None:
        return standard_fm_mse(forward(model,prepared,prepared['noisy'],prepared['sigma']),prepared['target'],prepared['noise'])
    losses=[]
    generative=(prepared['tasks']==0)|(prepared['tasks']==2)
    if generative.any():
        p=subset(prepared,generative)
        pred=unpack(forward(model,p,p['noisy'],p['sigma']),p['target'])
        losses.append(((pred.float()-(p['noise'].float()-p['target'].float()))**2).flatten(1).mean(1))
    if (~generative).any():
        p=subset(prepared,~generative)
        velocity=lambda state,sigma:unpack(forward(model,p,state,sigma.to(state.dtype)),p['target'])
        reached,_=detached_rollout_to_k(initial_noise=p['noise'],k=k,velocity_fn=velocity)
        sigma=torch.full((reached.shape[0],),SIGMAS[k],device=reached.device,dtype=reached.dtype)
        target=corrective_velocity_target(reached,p['target'],sigma)
        pred=unpack(forward(model,p,reached,sigma),p['target'])
        losses.append(((pred.float()-target.float())**2).flatten(1).mean(1))
    return torch.cat(losses).mean()
