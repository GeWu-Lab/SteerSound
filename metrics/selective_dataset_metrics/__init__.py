"""Selective evaluation: lazy metric models, portable cache paths."""
from importlib import import_module
from pathlib import Path
import os
DEFAULT_CLAP=Path(os.environ.get('STEERSOUND_CLAP_PATH',str(Path.home()/'.cache/steersound/clap'))).expanduser()
DEFAULT_TORCH_HOME=Path(os.environ.get('TORCH_HOME',str(Path.home()/'.cache/torch'))).expanduser()
__all__=['remove_drop_db', 'preserve_drift_db', 'volume_change_db', 'volume_progress', 'pitch_shift_semitone', 'pitch_progress', 'clap_similarity', 'generate_completion', 'add_completion', 'remove_completion', 'replace_completion', 'volume_completion', 'pitch_completion', 'summarize_completion', 'summarize_tasks', 'summarize_classifications']
_AGGREGATION=['summarize_completion', 'summarize_tasks', 'summarize_classifications']
def __getattr__(name):
 if name not in __all__:raise AttributeError(name)
 value=getattr(import_module(f'{__name__}.'+('aggregation' if name in _AGGREGATION else 'metrics')),name)
 globals()[name]=value
 return value
