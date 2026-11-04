"""Complex evaluation: lazy metric models, portable cache paths."""
from importlib import import_module
from pathlib import Path
import os
DEFAULT_CLAP=Path(os.environ.get('STEERSOUND_CLAP_PATH',str(Path.home()/'.cache/steersound/clap'))).expanduser()
DEFAULT_TORCH_HOME=Path(os.environ.get('TORCH_HOME',str(Path.home()/'.cache/torch'))).expanduser()
__all__=['volume_mae_db', 'lsd', 'pitch_mae_semitone', 'clap_task_score', 'kl_softmax', 'summarize_metric', 'summarize_tasks', 'tbg', 'tbg_all', 'tbg_selective']
_AGGREGATION=['summarize_metric', 'summarize_tasks', 'tbg', 'tbg_all', 'tbg_selective']
def __getattr__(name):
 if name not in __all__:raise AttributeError(name)
 value=getattr(import_module(f'{__name__}.'+('aggregation' if name in _AGGREGATION else 'metrics')),name)
 globals()[name]=value
 return value
