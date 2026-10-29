# Pretraining and fine-tuning with standard FM

Use the original audio datasets used in our paper, or your own audio–caption data. After preparation, the loader constructs editing instructions and input/target audio online. Pretraining and ordinary fine-tuning both use standard Flow Matching with Logit-Normal noise sampling; validation and final evaluation use **Complex audio editing data**.

| Mode | Entry point | Editor initialization |
|---|---|---|
| Pretraining | `train.py` | New DiT. |
| Standard FM fine-tuning | `finetune.py` | Existing editor supplied with `--dit`. |

The DiT is trainable; the downloaded VAE, text encoder, and vocoder remain frozen. For Emphasis or Recovery, follow the separate [STEER post-training guides](../STEER_post_training/README.md).

## 1. Prepare audio, weights, and evaluation data

Run from the **repository root**, except where a `cd` is shown. Follow [Installation](../README.md#installation), then install:

```bash
pip install -r requirements-training.txt
pip install -r requirements-metrics.txt
```

**Training audio:** follow [Source audio and preparation](dataset/README.md) for the paper's source audio or your own audio–caption data. The guide contains the original dataset links, input formats, timing annotation, CLAP filtering, data checks, and expected outputs.

**Expected training input:** `data/my_dataset/prepared/online_data.json` plus the source audio and metadata it references. You supply source audio and text; editing input/target pairs are constructed online during training.

Download the model components and Complex validation/test data:

```python
from huggingface_hub import snapshot_download

snapshot_download("GeWu-Lab/SteerSound", local_dir="weights/SteerSound")
snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["complex_val/*", "complex_test/*"],
)
```

**Expected files:**

```text
weights/SteerSound/              # editor and frozen components
data/
├── my_dataset/prepared/online_data.json
├── complex_val/
│   ├── metadata.parquet
│   ├── audio.zip                # input and GT target WAVs
│   └── manifest.json
└── complex_test/                # same three files
```

Keep the ZIPs zipped. Prepare the [metric dependencies](../STEER_post_training/README.md#shared-setup) before evaluation. Complex val/test is used for evaluation; the online training loader reads the audio referenced by your config.

## 2. Pretrain a new editor

From the repository root:

```bash
cd pretrain_and_standard_FM_finetune

torchrun --standalone --nproc_per_node=1 train.py \
  --model ../weights/SteerSound \
  --online-config ../data/my_dataset/prepared/online_data.json \
  --epochs 10 --save-every 5 \
  --batch-size 1 --workers 2 \
  --output outputs/my_pretrain
```

This initializes a new DiT. Pretraining saves epoch checkpoints and runs without in-loop validation. After training, use [step 4](#4-evaluate-on-complex) to evaluate an exported checkpoint, such as `pretrain_and_standard_FM_finetune/outputs/my_pretrain/dit_epoch010.safetensors`, on Complex val/test. Adjust the epoch count for your training budget.

## 3. Standard fine-tuning

This loads an existing editor and updates it with the same standard Flow Matching objective. Complex validation selects the editor; test data is reserved for final evaluation.

First measure the initialization checkpoint on Complex validation. Run from the repository root:

```bash
python inference/evaluate.py \
  --model weights/SteerSound \
  --dit weights/SteerSound/dit/model.safetensors \
  --data data/complex_val \
  --setting complex --split val \
  --output outputs/standard_fm_initial_complex_val
```

**Expected result:** `outputs/standard_fm_initial_complex_val/summary.json`. The Complex selection score compares validation metrics with this starting checkpoint; pass the summary as `--baseline-val`. If initializing from your own editor, use that same `--dit` in both commands.

Then fine-tune:

```bash
cd pretrain_and_standard_FM_finetune

torchrun --standalone --nproc_per_node=1 finetune.py \
  --model ../weights/SteerSound \
  --dit ../weights/SteerSound/dit/model.safetensors \
  --online-config ../data/my_dataset/prepared/online_data.json \
  --val ../data/complex_val \
  --baseline-val ../outputs/standard_fm_initial_complex_val/summary.json \
  --epochs 3 --batch-size 1 --workers 2 \
  --output outputs/my_finetune
```

Use the validation-selected editor in step 4. Validation is optional: omit both `--val` and `--baseline-val` to train for a fixed budget, then use `final.safetensors` from the output folder.

## 4. Evaluate on Complex

Run from the **repository root**. After fine-tuning, return there with `cd ..`.

Evaluate the selected fine-tuned editor on Complex validation:

```bash
python inference/evaluate.py \
  --model weights/SteerSound \
  --dit pretrain_and_standard_FM_finetune/outputs/my_finetune/selected.safetensors \
  --data data/complex_val \
  --setting complex --split val \
  --output outputs/standard_fm_complex_val
```

Then evaluate the same editor on Complex test:

```bash
python inference/evaluate.py \
  --model weights/SteerSound \
  --dit pretrain_and_standard_FM_finetune/outputs/my_finetune/selected.safetensors \
  --data data/complex_test \
  --setting complex --split test \
  --output outputs/standard_fm_complex_test
```

For pretraining, replace `--dit` in both commands with the exported epoch checkpoint you want to evaluate. Use validation to compare candidate checkpoints and evaluate the chosen checkpoint on test afterward. For a fine-tuning run without validation, use its `final.safetensors` instead.

**Expected output:** `summary.json` and `per_sample.parquet` in each output directory, containing the applicable Complex metrics for all six tasks. The evaluator reads reference targets directly from the downloaded ZIPs. See [evaluation fields and audio requirements](../inference/evaluation/README.md#complex-data).

## 5. Use the fine-tuned editor

From the repository root:

```bash
python inference/infer.py \
  --model weights/SteerSound \
  --dit pretrain_and_standard_FM_finetune/outputs/my_finetune/selected.safetensors \
  --input input.wav \
  --instruction "Remove the sound of a dog barking." \
  --output outputs/finetune_demo.wav
```

Set process count, batch size, learning rate (`--lr`), and warmup (`--warmup`) for your setup. Use a new output directory per run. Run `python train.py --help` or `python finetune.py --help` from this training folder for options.

## Layout

```text
pretrain_and_standard_FM_finetune/
├── train.py
├── finetune.py
├── online_data.example.json
├── dataset/
│   ├── dataset.py
│   ├── paired.py
│   ├── balanced_sampler.py
│   ├── worker_determinism.py
│   ├── preparation/       # timing annotation, conversion, preflight
│   ├── examples/          # audio and manifests for both source types
│   └── create_example_data.py
├── utils/
│   ├── trainer.py
│   └── ...
├── steersound/
│   └── models/
└── metrics/
```

Data loading and preparation are grouped in `dataset/`, alongside the training entry point.
