# Continue training SteerSound

Fine-tune the released [🤗 SteerSound checkpoint](https://huggingface.co/GeWu-Lab/SteerSound) for downstream audio editing using Emphasis or Recovery.

## What is available on Hugging Face?

| Released asset | Contents | Use |
|---|---|---|
| [🤗 Model weights](https://huggingface.co/GeWu-Lab/SteerSound) | DiT editor, audio VAE, text encoder/tokenizer, and vocoder. | Initialize continued training and inference. |
| Complex audio editing data | `complex_val`, `complex_test`, including input audio, instructions, and GT targets. | Validation and final evaluation. |
| [🤗 Selective audio editing data](https://huggingface.co/datasets/GeWu-Lab/SteerSound) | `selective_train` with editing pairs; `selective_val` and `selective_test` with inputs and task annotations. | Training, validation, and final evaluation. |

## Choose a training workflow

| Workflow | Training data | Validation / test |
|---|---|---|
| **1. Complex audio editing data** | Follow [Source audio and preparation](../pretrain_and_standard_FM_finetune/dataset/README.md) to prepare the paper's source audio or your own audio–caption data. The loader constructs editing examples online for post-training. | Use `complex_val` for validation and `complex_test` for final evaluation. |
| **2. Selective audio editing data** | Download our released [🤗 Selective audio editing data](https://huggingface.co/datasets/GeWu-Lab/SteerSound) and train directly on its editing pairs in `selective_train`. | Use `selective_val` and `selective_test`. |

Choose Emphasis or Recovery for either workflow:

| Method | Complex audio editing data | Selective audio editing data |
|---|---|---|
| Emphasis | [Prepare data and train](emphasis/complex/README.md) | [Download and train](emphasis/selective/README.md) |
| Recovery | [Prepare data and train](recovery/complex/README.md) | [Download and train](recovery/selective/README.md) |

```text
1. Source audio: paper datasets or your own audio–caption data
   → online editing-example construction → post-training
   → validate on complex_val → evaluate on complex_test

2. Released selective_train editing pairs → post-training
   → validate on selective_val → evaluate on selective_test
```

## Shared setup

Follow [Installation](../README.md#installation). From the repository root:

```bash
pip install -r requirements-training.txt
pip install -r requirements-metrics.txt
```

Validation uses the following dependencies. Listing a manual CLAP download does **not** mean the evaluator only computes CLAP.

| Metric | How it is computed | Weight preparation |
|---|---|---|
| Task-aware CLAP | CLAP audio/text similarities for the task's labels. | Download `laion/larger_clap_general` using the command below; override its location with `STEERSOUND_CLAP_PATH`. |
| KL (Complex) | Compare PaSST class-probability distributions for generated and target audio. | The retained PaSST factory automatically downloads pretrained weights on first use and reuses the Torch cache; `TORCH_HOME` can change the cache root. |
| LSD (Complex) | Log-spectral distance to the target. | No pretrained weights. |
| Volume MAE (Complex) | Absolute RMS level difference in dB against the target. | No pretrained weights. |
| Pitch MAE (Complex) | Paired multi-resolution CQT pitch-shift measurement against the target, in semitones. | No pretrained weights. |
| Selective success/error classification | Frozen task rules using CLAP, level changes, or CQT pitch changes. | Uses CLAP where required and signal-processing functions; no separate success-classifier weights. |

Download the CLAP checkpoint:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "laion/larger_clap_general",
    local_dir=Path.home() / ".cache/steersound/clap",
)
```

Set `STEERSOUND_CLAP_PATH` if you keep it elsewhere. This is the CLAP checkpoint used by validation; source-audio time annotation uses its own checkpoint, as explained in the [data guide](../pretrain_and_standard_FM_finetune/dataset/README.md).

## Reported metrics and checkpoint selection

The Complex evaluator computes the following metrics in one run:

| Tasks | Reported metrics |
|---|---|
| Generate, Add, Replace | Task-aware CLAP and KL. |
| Remove | Task-aware CLAP, LSD, and KL. |
| Pitch | Pitch MAE, LSD, and KL. |
| Volume | Volume MAE, LSD, and KL. |

In `summary.json`, `tasks.<task>.metrics` contains the task's metric means and valid counts. `primary` contains the six values used for normalization: CLAP for Generate/Add/Replace/Remove and the corresponding MAE for Pitch/Volume. KL and LSD are reported alongside them.

**Why measure the starting checkpoint first?** The current Complex training runner selects checkpoints using `TBG_all`, which combines improvement ratios relative to the initialization checkpoint on the same validation set. That requires a reference `summary.json` once per initialization/split. It does not require another model or another set of metric weights. The reference evaluation itself already computes the individual metrics listed above.

For standalone evaluation, `--baseline` is optional: without it, the evaluator reports absolute metrics; supplying it adds TBG summaries. Complex training requires `--baseline-val` because its checkpoint selection uses `TBG_all`. Selective training instead selects by overall success rate and does not require this reference-evaluation step.

## Which data path should I use?

- **Complex audio editing data:** use [Source audio and preparation](../pretrain_and_standard_FM_finetune/dataset/README.md) for the paper's source audio or your own audio–caption data. Pass `online_data.json` as `--online-config` for online construction; use `complex_val` / `complex_test` for validation and final evaluation.
- **Selective audio editing data:** download `selective_train`, `selective_val`, and `selective_test` from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound). Keep their `audio.zip` files zipped. The Selective guides give the exact download and directory layout.
- **Standard pretraining or fine-tuning:** use [pretrain_and_standard_FM_finetune/](../pretrain_and_standard_FM_finetune/README.md). That route uses standard Flow Matching; the guides on this page apply STEER.

## Download Complex validation and test data

Run this from the repository root to download both public splits from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["complex_val/*", "complex_test/*"],
)
```

**Expected files:**

```text
data/
├── complex_val/
│   ├── metadata.parquet
│   ├── audio.zip
│   └── manifest.json
└── complex_test/             # same three filenames
```

Keep the ZIPs zipped. These folders contain **input audio, instructions, and ground-truth target audio**, ready for Complex validation and test metrics. Use `data/complex_val` for checkpoint selection and `data/complex_test` for final evaluation.

**Test is optional** and is evaluated after selection. You can also use your own paired validation/test collection in the [same format](emphasis/complex/dataset/README.md#prepare-complex-data).

## Continue from your own checkpoint

The examples set `--dit` to `weights/SteerSound/dit/model.safetensors`. To continue from a model you trained, change it to your `selected.safetensors`, keeping `--model` pointed at the model components. For example, after Emphasis you can initialize Recovery with its selected editor.

For Complex, recompute the validation reference with the **same editor used for initialization** by adding the corresponding `--dit` to the evaluation command and using a new output directory. Pass its `summary.json` to `--baseline-val`. Any test reference must be measured on its own test split.

## Directory layout

Each `emphasis/{complex,selective}` and `recovery/{complex,selective}` folder contains `train.py`, `dataset/`, `utils/`, `steersound/`, and `metrics/`. Data loading, batch sampling, and worker setup live in that entry point's top-level `dataset/`.
