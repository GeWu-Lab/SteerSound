"""Read published metadata and immutable WAV members directly from audio.zip."""
from __future__ import annotations
import ast,io,json,os,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from steersound.instructions import strip_model_prefix

class AudioRecords:
    def __init__(self,folder,*,split=None,paired=False):
        self.folder=Path(folder).expanduser().resolve()
        self.frame=pd.read_parquet(self.folder/'metadata.parquet')
        required={'sample_id','split','task','task_id','instruction','input_audio_path','target_audio_path','change_vol','change_Pitch'}
        if not required.issubset(self.frame):raise ValueError(f'Missing metadata columns: {required-set(self.frame)}')
        if self.frame.empty or self.frame.sample_id.duplicated().any():raise ValueError('Empty or duplicate sample IDs')
        if split and set(self.frame['split'])!={split}:raise ValueError(f'Expected {split} split')
        if paired and not self.frame.target_audio_path.map(lambda p:isinstance(p,str) and bool(p)).all():
            raise ValueError('Paired targets are required for supervised training and full Complex reference metrics. Public val/test splits omit target audio; provide a private paired dataset for these operations.')
        self._zip=None;self._pid=None
    def __len__(self):return len(self.frame)
    def __getstate__(self):
        state=self.__dict__.copy();state.update(_zip=None,_pid=None);return state
    def wave(self,member):
        if self._zip is None or self._pid!=os.getpid():
            self._zip=zipfile.ZipFile(self.folder/'audio.zip');self._pid=os.getpid()
        if not isinstance(member,str) or not member or Path(member).is_absolute() or '..' in Path(member).parts:
            raise ValueError('Invalid archive member path')
        wav,sr=sf.read(io.BytesIO(self._zip.read(member)),dtype='float32')
        if sr!=16000 or wav.shape!=(160000,) or not np.isfinite(wav).all():
            raise ValueError(f'Expected a finite 16 kHz mono 10-second WAV: {member}')
        return wav

class PairedDataset(Dataset):
    def __init__(self,folder):self.records=AudioRecords(folder,split='train',paired=True)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records.frame.iloc[index]
        ref=np.zeros(160000,dtype=np.float32) if int(row.task_id)==2 else self.records.wave(row.input_audio_path)
        return dict(prompt_text=strip_model_prefix(row.instruction),ref_audio=torch.from_numpy(ref.copy())[None],
                    tar_audio=torch.from_numpy(self.records.wave(row.target_audio_path).copy())[None],
                    task=torch.tensor(int(row.task_id)),change_vol=torch.tensor(float(row.change_vol)),
                    change_Pitch=torch.tensor(float(row.change_Pitch)))

def labels(value):
    if value is None:return []
    if isinstance(value,str):
        if not value.strip():return []
        if value.startswith(('[','(')):
            parsed=ast.literal_eval(value)
            if isinstance(parsed,(list,tuple)):return [str(x) for x in parsed]
        return [value]
    return [str(x) for x in value]

def region(row,prefix):
    a,b=int(row[prefix+'_start_sample']),int(row[prefix+'_end_sample'])
    if not 0<=a<b<=160000:raise ValueError(f'Invalid {prefix} region for {row["sample_id"]}: {a,b}')
    return a/16000,b/16000
