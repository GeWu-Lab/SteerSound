# Fine-tune SteerSound — Emphasis / Selective audio editing data

This guide applies Emphasis to the released [SteerSound Selective audio editing data](https://huggingface.co/datasets/GeWu-Lab/SteerSound), starting from the [🤗 SteerSound checkpoint](https://huggingface.co/GeWu-Lab/SteerSound).

Follow the [environment and metric setup](../../README.md#shared-setup) first. Run preparation commands from the **repository root**; the training command explicitly enters this folder.

## What you need

| Input | Where to get it | Expected location |
|---|---|---|
| SteerSound checkpoint and frozen components | [🤗 Model weights](https://huggingface.co/GeWu-Lab/SteerSound) | `weights/SteerSound/`, including `dit/model.safetensors`. |
| Selective training, validation, and test splits | [🤗 Dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound) | `data/selective_train`, `data/selective_val`, `data/selective_test`. |
| Evaluation dependencies | [Metric setup](../../README.md#shared-setup) | CLAP download plus signal-processing dependencies for success/error rules. |

The training split contains prepared editing pairs. Validation selects the editor; test is evaluated after selection. This example uses `--train` to read the released pairs directly.

## 1. Download weights and data

**Why:** place every required asset at the paths used by the training command.

```python
from huggingface_hub import snapshot_download

snapshot_download("GeWu-Lab/SteerSound", local_dir="weights/SteerSound")
snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["selective_train/*", "selective_val/*", "selective_test/*"],
)
```

**Expected files:**

```text
weights/SteerSound/
├── config.json
├── dit/model.safetensors
└── ...                       # VAE, text encoder, and vocoder

data/
├── selective_train/
│   ├── metadata.parquet
│   ├── audio.zip
│   └── manifest.json
├── selective_val/            # same three filenames
└── selective_test/           # same three filenames
```

Keep `audio.zip` zipped; the loader reads its WAV members directly. See [split contents](dataset/README.md#released-selective-data) for the record counts and roles.

## 2. Continue training

**Why:** initialize the editor from the released weights and update it on Selective training data. `--model` supplies the model components; `--dit` identifies the editor to initialize.

```bash
cd STEER_post_training/emphasis/selective

torchrun --standalone --nproc_per_node=1 train.py \
  --model ../../../weights/SteerSound \
  --dit ../../../weights/SteerSound/dit/model.safetensors \
  --train ../../../data/selective_train \
  --val ../../../data/selective_val \
  --test ../../../data/selective_test \
  --epochs 3 \
  --batch-size 1 \
  --workers 2 \
  --output outputs/continue_emphasis_selective
```

For STEER with online construction from your audio–caption pairs and evaluation on Complex, follow the [audio–caption workflow](../complex/README.md). For ordinary Flow Matching, follow [standard fine-tuning](../../../pretrain_and_standard_FM_finetune/README.md#3-standard-fine-tuning).

## 3. Use the selected editor

**Expected output:** `outputs/continue_emphasis_selective/selected.safetensors` in this training folder. Validation chooses this editor; the starting checkpoint remains eligible if later updates do not improve the score.

Return to the repository root and run:

```bash
cd ../../..

python inference/infer.py \
  --model weights/SteerSound \
  --dit STEER_post_training/emphasis/selective/outputs/continue_emphasis_selective/selected.safetensors \
  --input input.wav \
  --instruction "Remove the sound of a dog barking." \
  --output outputs/continue_emphasis_selective_demo.wav
```

**Expected result:** a mono 16 kHz, 10-second edited WAV. Use a new `--output` training directory for each run. Set process count, batch size, learning rate (`--lr`), and warmup (`--warmup`) for your training setup. Run `python train.py --help` from this training folder for options.

For the validation/test file schema and standalone evaluation command, see [evaluation data requirements](../../../inference/evaluation/README.md) and [run evaluation](../../../inference/README.md#evaluate).

## Directory layout

```text
selective/
├── train.py
├── dataset/
│   ├── dataset.py              # online construction from source audio
│   ├── paired.py               # prepared editing pairs and evaluation records
│   ├── balanced_sampler.py    # six-task batch sampling
│   └── worker_determinism.py  # reproducible DataLoader workers
├── utils/                     # training loop, objectives, and evaluation
├── steersound/                # model and audio/text components
└── metrics/                   # evaluation metrics
```

Each training entry point keeps its own implementation. See [dataset/](dataset/README.md) for the local loaders.
