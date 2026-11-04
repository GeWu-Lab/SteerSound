#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Evaluate a complete published validation or test split")
    p.add_argument('--model',default='GeWu-Lab/SteerSound');p.add_argument('--dit',type=Path)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--setting',required=True,choices=['complex','selective'])
    p.add_argument('--split',required=True,choices=['val','test']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--metric-device',default='cpu')
    a=p.parse_args()
    from evaluation.data import AudioRecords
    AudioRecords(a.data,split=a.split,paired=a.setting=='complex')
    import torch
    from safetensors.torch import load_file
    from steersound import SteerSoundPipeline
    from evaluation.evaluate import evaluate
    pipe=SteerSoundPipeline.from_pretrained(a.model,device=a.device)
    if a.dit:pipe.dit.load_state_dict(load_file(str(a.dit)),strict=True)
    baseline=json.loads(a.baseline.read_text())['primary'] if a.baseline else None
    report=evaluate(pipe,a.data,a.setting,a.output,split=a.split,baseline=baseline,metric_device=a.metric_device)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
