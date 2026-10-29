#!/usr/bin/env python3
"""Edit audio by supplying a plain-language instruction."""
from pathlib import Path
import argparse

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--instruction',required=True);p.add_argument('--input',type=Path)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='GeWu-Lab/SteerSound')
 p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['bfloat16','float16','float32'],default='bfloat16')
 p.add_argument('--seed',type=int,default=20260804)
 a=p.parse_args()
 from steersound.instructions import parse_instruction
 spec=parse_instruction(a.instruction)
 if spec.task!='generate' and a.input is None:p.error('--input is required for editing tasks')
 import torch,soundfile as sf
 from steersound import SteerSoundPipeline
 pipe=SteerSoundPipeline.from_pretrained(a.model,device=a.device,dtype=getattr(torch,a.precision))
 wave=pipe(a.instruction,audio=a.input,seed=a.seed)
 a.output.parent.mkdir(parents=True,exist_ok=True);sf.write(str(a.output),wave,16000,subtype='PCM_16')
 print(f'Saved {a.output}; task={spec.task}, volume={spec.change_vol:g} dB, pitch={spec.change_pitch:g} semitones')
if __name__=='__main__':main()
