# Metrics

The standalone packages accept one waveform/file at a time. Import `metrics.complex_dataset_metrics` or `metrics.selective_dataset_metrics` from the repository root. Install the root requirements-metrics.txt after CUDA PyTorch. Metrics also work on CPU. All indices, file iteration and checkpoint selection live in the evaluation/experiment programs.

Complex exports only task-aware CLAP, softmax KL(target || generated), pitch/volume MAE, LSD and TBG aggregation. Selective exports completion, its required features and frozen error-classification rules. Its pitch measurability failures are excluded from the accuracy denominator.

## Metric dependencies

CLAP is the manually downloaded scoring model. Complex KL also uses a pretrained **PaSST** model, which the retained factory downloads automatically on first use and caches under Torch's cache root (`TORCH_HOME` can override it). LSD, RMS volume error, and paired-CQT pitch error are signal-processing calculations and have no pretrained weights. Selective success/error rules reuse CLAP or those signal measurements; they have no separate classifier checkpoint.

For the complete dependency table and CLAP command, see [metric setup](../STEER_post_training/README.md#shared-setup). For the difference between absolute metrics, reference scores, and the Complex `TBG_all` selection score, see [reported metrics and selection](../STEER_post_training/README.md#reported-metrics-and-checkpoint-selection).

```python
from metrics.complex_dataset_metrics import lsd, volume_mae_db, tbg_all
print(lsd("edited.wav", "target.wav"))
print(volume_mae_db("edited.wav", "target.wav", region=(2., 4.)))
from metrics.selective_dataset_metrics import remove_completion
success = remove_completion("edited.wav", "input.wav", edit_region=(2.,4.), preserve_region=(6.,8.))
```

Regions use seconds at the resampled 16 kHz rate. Metric readers preserve duration; the Selective protocol explicitly normalizes to 10 seconds. All arrays must be finite mono waveforms. Task-aware CLAP takes base_labels/add_labels/remove_labels. Download the original `laion/larger_clap_general` via huggingface_hub.snapshot_download into ~/.cache/steersound/clap, or set STEERSOUND_CLAP_PATH. PaSST for softmax KL downloads through Torch Hub; TORCH_HOME overrides the cache root.

`TBG_all=100*(geomean(task-oriented ratios)-1)`; CLAP ratios are current/base and MAE ratios are base/current. TBG_sel retains Remove/Replace/Pitch/Volume. Compute task means from valid rows first, with equal weights across tasks; do not average per-sample or per-rank TBG. Validation and Test must use their corresponding base summaries. The original internal TBRS score is the same geometric ratio before subtracting 1 and scaling by 100, so checkpoint ordering is unchanged.

Every experiment contains its own local metric copy. Complex programs include only the Complex package; Selective programs include only the Selective package. Inference supports both protocols.
