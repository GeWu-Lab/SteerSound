"""GPU inference using the exact released DiT, audio VAE, T5 and vocoder."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import T5EncoderModel,T5Tokenizer,SpeechT5HifiGan
from .configuration import Edit_FluxParams,AutoEncoderParams
from .models import Edit_Flux,AutoEncoder
from .instructions import parse_instruction,model_prompt
from .audio import load_audio,SAMPLE_RATE
from .codec import vae_encode,vae_decode
from .sampling import BranchInputs,inference_dit_explicit


def _seed(base,sample_id,stream):
    return int.from_bytes(hashlib.sha256(f'{base}\0{sample_id}\0{stream}'.encode()).digest()[:8],'big')%(2**31)


def _load_model(cls,params,path,device,dtype):
    with torch.device('meta'):model=cls(params)
    model.load_state_dict(load_file(str(path),device='cpu'),strict=True,assign=True)
    return model.to(device=device,dtype=dtype).eval().requires_grad_(False)


class SteerSoundPipeline:
    sample_rate=SAMPLE_RATE
    def __init__(self,dit,vae,text_encoder,tokenizer,vocoder,device,dtype):
        self.dit=dit;self.vae=vae;self.text_encoder=text_encoder;self.tokenizer=tokenizer
        self.vocoder=vocoder;self.device=torch.device(device);self.dtype=dtype
        self.last_instruction=None

    @classmethod
    def from_pretrained(cls,model_id='GeWu-Lab/SteerSound',*,device='cuda',dtype=torch.bfloat16,revision=None):
        root=Path(model_id).expanduser()
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root=Path(snapshot_download(repo_id=model_id,revision=revision))
        config=json.loads((root/'config.json').read_text())
        if config['format_version']!=1:raise ValueError('Unsupported model format version')
        if torch.device(device).type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for GPU inference. Install a CUDA-enabled PyTorch build and use an NVIDIA GPU.')
        dit=_load_model(Edit_Flux,Edit_FluxParams(**config['dit']),root/'dit/model.safetensors',device,dtype)
        vae=_load_model(AutoEncoder,AutoEncoderParams(**config['autoencoder']),root/'vae/model.safetensors',device,dtype)
        tokenizer=T5Tokenizer.from_pretrained(str(root/'text_encoder'),local_files_only=True)
        encoder=T5EncoderModel.from_pretrained(str(root/'text_encoder'),torch_dtype=dtype,local_files_only=True).to(device).eval().requires_grad_(False)
        vocoder=SpeechT5HifiGan.from_pretrained(str(root/'vocoder'),torch_dtype=dtype,local_files_only=True).to(device).eval().requires_grad_(False)
        return cls(dit,vae,encoder,tokenizer,vocoder,device,dtype)

    @torch.inference_mode()
    def encode_text(self,instructions):
        # Control tokens belong to the model; empty CFG conditioning stays empty.
        inputs=self.tokenizer([model_prompt(t) for t in instructions],max_length=128,truncation=True,padding=True,return_tensors='pt')
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        hidden=self.text_encoder(**inputs,return_dict=True).last_hidden_state.to(self.dtype)
        return hidden,inputs['attention_mask'].bool()

    @torch.inference_mode()
    def __call__(self,instruction,audio=None,*,sample_rate=None,seed=20260804):
        spec=parse_instruction(instruction)
        if audio is None and spec.task!='generate':raise ValueError('Input audio is required for editing tasks')
        if spec.task=='generate':audio=None
        return self.edit_record(spec.conditioning_text,audio,sample_rate=sample_rate,seed=seed,
                                task_id=spec.task_id,change_vol=spec.change_vol,change_pitch=spec.change_pitch)

    @torch.inference_mode()
    def edit_record(self,instruction,audio,*,sample_rate=None,seed=20260804,sample_id='single',task_id,change_vol=0.,change_pitch=0.,cfg_scale=None):
        """Evaluation adapter: retain exact scalar metadata from a published index."""
        if int(task_id) not in range(6):raise ValueError('Unknown task id')
        if abs(float(change_vol))>5 or abs(float(change_pitch))>4:raise ValueError('Change exceeds the checkpoint training range')
        reference=load_audio(None if int(task_id)==2 else audio,sample_rate).to(self.device)
        generator=torch.Generator(device=self.device).manual_seed(_seed(seed,sample_id,'vae'))
        encoded=vae_encode({'ref_audio':reference,'tar_audio':reference.clone()},self.vae,SimpleNamespace(duration=10.),
                           self.device,self.dtype,generator=generator)['ref_audio']
        text,mask=self.encode_text([instruction]);null_text,null_mask=self.encode_text([''])
        task=torch.tensor([int(task_id)],device=self.device)
        volume=torch.tensor([float(change_vol)],device=self.device,dtype=torch.float32)
        pitch=torch.tensor([float(change_pitch)],device=self.device,dtype=torch.float32)
        cond=BranchInputs(encoded,text,mask,task,volume,pitch)
        null=BranchInputs(encoded,null_text,null_mask,task,volume,pitch)
        generator=torch.Generator(device=self.device).manual_seed(_seed(seed,sample_id,'diffusion'))
        noise=torch.randn(encoded.shape,generator=generator,device=self.device,dtype=self.dtype)
        guidance=float(cfg_scale) if cfg_scale is not None else 8.5 if int(task_id)==0 else 3.5
        latent,_=inference_dit_explicit(self.dit,cond,noise,cfg_scale=guidance,null=null,
                                       device=str(self.device),dtype=self.dtype,num_steps=50,collect_trace=False)
        wave=np.asarray(vae_decode(latent,self.vae,self.vocoder)[0],dtype=np.float32)
        if wave.shape!=(160000,) or not np.isfinite(wave).all():raise RuntimeError('Invalid generated waveform')
        return wave
