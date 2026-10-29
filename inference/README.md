# Inference and evaluation

Follow [Installation](../README.md#installation). Run commands from the **repository root**.

## Edit audio

```bash
python inference/infer.py \
  --model GeWu-Lab/SteerSound \
  --input input.wav \
  --instruction "Remove the sound of a dog barking." \
  --output edited.wav
```

## Generate audio

```bash
python inference/infer.py \
  --model GeWu-Lab/SteerSound \
  --instruction "Generate the sound of rain." \
  --output rain.wav
```

Use a local downloaded directory as `--model` if preferred. To use an editor you trained, add `--dit /path/to/editor.safetensors`; the frozen components come from `--model`. See [instruction examples](../README.md#inference) for all six tasks.

## Evaluate

### 1. Download and prepare evaluation data

Both settings download their published evaluation data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound). Choose the setting you will evaluate.

**Selective**

Download validation/test data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["selective_val/*", "selective_test/*"],
)
```

**Expected folders:** `data/selective_val` and `data/selective_test`. The published data supports Selective success/error evaluation directly, without reference-target audio.

**Complex**

Download validation/test data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["complex_val/*", "complex_test/*"],
)
```

**Expected folders:** `data/complex_val` and `data/complex_test`. These splits include input audio, instructions, and ground-truth target audio, ready for full Complex evaluation. For your own paired collection, follow the [Complex format](evaluation/README.md#complex-data).

Each downloaded folder contains `metadata.parquet`, `audio.zip`, and `manifest.json`. Both settings use mono 16 kHz, 10-second WAV members. Keep the ZIP zipped. Cover all six tasks and match the table's split to `--split`. See [field definitions and examples](evaluation/README.md).

### 2. Prepare metric dependencies

```bash
pip install -r requirements-metrics.txt
```

```python
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "laion/larger_clap_general",
    local_dir=Path.home() / ".cache/steersound/clap",
)
```

Set `STEERSOUND_CLAP_PATH` to use a different CLAP directory. Complex KL uses pretrained PaSST weights, which download automatically on first use (`TORCH_HOME` controls the cache root). LSD and Pitch/Volume MAE use signal processing and need no model weights. See [metric dependencies](../metrics/README.md#metric-dependencies).

### 3. Evaluate a split

Selective test:

```bash
python inference/evaluate.py \
  --model GeWu-Lab/SteerSound \
  --data data/selective_test \
  --setting selective --split test \
  --output outputs/selective_test
```

Complex test:

```bash
python inference/evaluate.py \
  --model GeWu-Lab/SteerSound \
  --data data/complex_test \
  --setting complex --split test \
  --output outputs/complex_test
```

Use `--split val` and the corresponding validation folder to evaluate validation data. Add `--dit /path/to/editor.safetensors` to evaluate a pretraining, standard fine-tuning, or STEER checkpoint. Use a new output directory for each run.

### 4. Read the results

**Expected output:**

```text
outputs/<run>/
├── summary.json
└── per_sample.parquet
```

`summary.json` contains per-task Complex metrics or per-task Selective success rates and overall success/error rates. `per_sample.parquet` contains the per-record measurements/classifications. A Complex `--baseline /path/to/reference_summary.json` additionally reports TBG; it is optional for standalone absolute metrics. The Complex STEER training guides use that reference for checkpoint selection.
