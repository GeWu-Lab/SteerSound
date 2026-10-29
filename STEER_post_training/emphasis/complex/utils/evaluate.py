"""Full-split evaluation; shard rows without replication and pool raw counts."""
from __future__ import annotations
import json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
from dataset.paired import AudioRecords,labels,region

TASKS=('add','remove','generate','replace','volume','pitch')
PRIMARY={t:('volume_mae_db' if t=='volume' else 'pitch_mae_semitone' if t=='pitch' else 'clap_task_score') for t in TASKS}

def _complex(row,out,target,device):
    from metrics import complex_dataset_metrics as m
    task=row['task']
    start,end=float(row.get('start_time',-1)),float(row.get('end_time',-1))
    bounds=(int(round(start*16000)),int(round(end*16000))) if 0<=start<end<=10 else None
    edit=(start,end) if bounds else None
    result={}
    if task in ('generate','add','remove','replace'):
        result['clap_task_score']=m.clap_task_score(out,task=task,base_labels=labels(row['base_labels']),
            add_labels=labels(row['add_labels']),remove_labels=labels(row['remove_labels']),region=edit,device=device)
    if task=='volume':result['volume_mae_db']=m.volume_mae_db(out,target,region=edit)
    if task=='pitch':result['pitch_mae_semitone']=m.pitch_mae_semitone(out,target,region=edit)
    if task in ('remove','volume','pitch'):result['lsd_edit']=m.lsd(out,target,region=edit)
    result['kl_edit']=m.kl_softmax(out,target,region=edit,device=device)
    if bounds is not None and bounds!=(0,160000):
        a,b=bounds
        out_keep=np.concatenate([out[:a],out[b:]])
        target_keep=np.concatenate([target[:a],target[b:]])
        result['lsd_unchanged']=m.lsd(out_keep,target_keep)
        result['kl_unchanged']=m.kl_softmax(out_keep,target_keep,device=device)
    return result

def _selective(row,out,ref,device):
    from metrics.selective_dataset_metrics import metrics as m
    from metrics.selective_dataset_metrics._rules import classify_row
    task=row['task'];features=dict(task=task)
    if task in ('add','generate'):
        target=row['target_label']
        features['clap_target_global']=m.clap_similarity(out,target,device=device)
    else:
        edit,preserve=region(row,'edit'),region(row,'preserve')
        if task=='replace':
            sim=lambda label,r:m.clap_similarity(out,label,region=r,device=device)
            features.update(clap_destination_edit=sim(row['destination_label'],edit),clap_source_edit=sim(row['source_label'],edit),
                clap_destination_preserve=sim(row['destination_label'],preserve),
                clap_base_preserve_output=min(sim(label,preserve) for label in labels(row['base_labels'])))
        elif task in ('remove','volume'):
            oa,ra=m._paired(out,ref,16000,edit);ob,rb=m._paired(out,ref,16000,preserve)
            features.update(output_preserve_db=m._rms(ob),reference_preserve_db=m._rms(rb))
            if task=='remove':features.update(remove_drop_db=m._rms(ra)-m._rms(oa),preserve_drift_db=abs(m._rms(ob)-m._rms(rb)))
            else:features.update(requested_change=float(row['change_vol']),edit_delta_db=m._rms(oa)-m._rms(ra))
        elif task=='pitch':
            a=m.pitch_shift_semitone(out,ref,region=edit);b=m.pitch_shift_semitone(out,ref,region=preserve)
            features.update(requested_change=float(row['change_Pitch']),edit_delta_semitone=a,preserve_delta_semitone=b,
                            output_edit_pitch_valid=math.isfinite(a),output_preserve_pitch_valid=math.isfinite(b))
    result=classify_row(SimpleNamespace(**features));result.update(features)
    result['completion']=float(result['metric_prediction']) if result['metric_in_denominator'] else float('nan')
    return result

def summarize(records,setting,baseline=None):
    if set(r['task'] for r in records)!=set(TASKS):raise ValueError('Full six-task evaluation is required')
    if setting=='complex':
        from metrics.complex_dataset_metrics.aggregation import summarize_tasks,tbg_all,tbg_selective
        names=('clap_task_score','volume_mae_db','pitch_mae_semitone','lsd_edit','lsd_unchanged','kl_edit','kl_unchanged')
        tasks=summarize_tasks(records,metric_names=names)
        primary={t:tasks[t]['metrics'][PRIMARY[t]]['mean'] for t in TASKS}
        report=dict(setting=setting,tasks=tasks,primary=primary)
        if baseline is not None:
            report.update(TBG_all=tbg_all(primary,baseline),TBG_sel=tbg_selective(primary,baseline))
    else:
        from metrics.selective_dataset_metrics.aggregation import summarize_tasks,summarize_classifications
        report=dict(setting=setting,tasks=summarize_tasks(records),overall=summarize_classifications(records))
    return report

@torch.inference_mode()
def evaluate(pipe,folder,setting,output,*,split,baseline=None,seed=20260804,metric_device='cpu'):
    records=AudioRecords(folder,split=split,paired=(setting=='complex'))
    rank=dist.get_rank() if dist.is_initialized() else 0;world=dist.get_world_size() if dist.is_initialized() else 1
    output=Path(output)
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    if dist.is_initialized():dist.barrier()
    was_training=pipe.dit.training;pipe.dit.eval();local=[]
    for index in range(rank,len(records),world):
        row=records.frame.iloc[index].to_dict()
        ref=np.zeros(160000,dtype=np.float32) if int(row['task_id'])==2 else records.wave(row['input_audio_path'])
        out=pipe.edit_record(row['instruction'],ref,sample_rate=16000,task_id=int(row['task_id']),
            change_vol=float(row['change_vol']),change_pitch=float(row['change_Pitch']),sample_id=str(row['sample_id']),seed=seed)
        # Formal metrics score exactly the PCM16 waveform written by native evaluation.
        out=(np.clip(np.floor(out.astype(np.float64)*32768),-32768,32767)/32768).astype(np.float32)
        scores=_complex(row,out,records.wave(row['target_audio_path']),metric_device) if setting=='complex' else _selective(row,out,ref,metric_device)
        local.append(dict(sample_id=row['sample_id'],task=row['task'],**scores))
    pd.DataFrame(local).to_parquet(output/f'rank_{rank:04d}.parquet',index=False)
    if dist.is_initialized():dist.barrier()
    report=None
    if rank==0:
        frame=pd.concat([pd.read_parquet(output/f'rank_{r:04d}.parquet') for r in range(world)],ignore_index=True)
        if len(frame)!=len(records) or frame.sample_id.duplicated().any() or set(frame.sample_id)!=set(records.frame.sample_id):
            raise RuntimeError('Evaluation coverage mismatch')
        # Restore source sample order, independent of rank count.
        frame=frame.set_index('sample_id').loc[records.frame.sample_id].reset_index()
        frame.to_parquet(output/'per_sample.parquet',index=False)
        report=summarize(frame.to_dict('records'),setting,baseline)
        report.update(split=split,rows=len(frame),selection_uses_test=False)
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if dist.is_initialized():
        shared=[report];dist.broadcast_object_list(shared,src=0);report=shared[0]
    pipe.dit.train(was_training)
    return report

def selection_score(report):
    score=report['TBG_all'] if report['setting']=='complex' else report['overall']['success_rate']
    if score is None or not math.isfinite(score):raise ValueError('Unmeasurable validation score')
    return float(score)
