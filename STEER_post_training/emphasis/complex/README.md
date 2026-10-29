# Fine-tune SteerSound — Emphasis / Complex audio editing data

Use Emphasis to post-train the released [🤗 SteerSound checkpoint](https://huggingface.co/GeWu-Lab/SteerSound) for Complex audio editing. Follow [Source audio and preparation](../../../pretrain_and_standard_FM_finetune/dataset/README.md) to prepare the paper's source audio or your own audio–caption data. Editing examples are constructed online; validation and final evaluation use Complex val/test.

Follow the [environment and metric setup](../../README.md#shared-setup) first. Run preparation commands from the **repository root**; the training command explicitly enters this folder.

## What you need

| Input | Where to get it | Expected location |
|---|---|---|
| SteerSound checkpoint and frozen components | [🤗 Model weights](https://huggingface.co/GeWu-Lab/SteerSound) | `weights/SteerSound/`, including `dit/model.safetensors`. |
| Training source audio | [Source audio and preparation](../../../pretrain_and_standard_FM_finetune/dataset/README.md): paper source audio or your own audio–caption data. | `data/my_dataset/prepared/online_data.json`. |
| Paired Complex validation data | [🤗 Dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound) / [Data format](dataset/README.md#prepare-complex-data) | `data/complex_val/metadata.parquet` and `audio.zip`. |
| Evaluation dependencies | [Metric setup](../../README.md#shared-setup) | CLAP download; PaSST auto-downloads for KL; LSD and MAE need no model weights. |

**Training:** prepared audio from the original datasets used in our paper or your own audio–caption data, with editing examples constructed online. **Evaluation:** `complex_val` and `complex_test`, including ground-truth target audio.

## 1. Download weights

```python
from huggingface_hub import snapshot_download

snapshot_download("GeWu-Lab/SteerSound", local_dir="weights/SteerSound")
```

**Expected result:** `weights/SteerSound/dit/model.safetensors` and its accompanying model components.

## 2. Prepare your data

**Training:** follow the [Source audio and preparation](../../../pretrain_and_standard_FM_finetune/dataset/README.md) through its data check. It prepares the audio, label/caption annotations, and sounding times needed by the online loader. Pass the resulting `data/my_dataset/prepared/online_data.json` as `--online-config` in the command below.

For validation and test, [download `complex_val` and `complex_test` from Hugging Face](../../README.md#download-complex-validation-and-test-data). They include input and ground-truth target audio at `data/complex_val` and `data/complex_test`. The training command below uses the released validation split; test data is optional. For your own paired validation data, follow the [same format](dataset/README.md#prepare-complex-data).

**Why:** source recordings provide online training examples; held-out references let the evaluator select an editor without using the training data.

**Expected files before proceeding:**

```text
data/
├── my_dataset/
│   └── prepared/online_data.json
├── complex_val/
│   ├── metadata.parquet
│   ├── audio.zip          # input and ground-truth target WAV members
│   └── manifest.json
└── complex_test/           # same three files; optional held-out test set
```

The config contains absolute paths to your source audio and timing metadata. Validation covers all six tasks and uses recordings held out from training.

## 3. Measure the starting checkpoint

**Why:** this fine-tuning workflow selects checkpoints using `TBG_all`, an aggregate of improvement ratios relative to the starting checkpoint on the **same validation split**. Here that starting checkpoint is the released SteerSound model. This reference measurement is required for the current training selection rule, not for computing individual absolute metrics.

This command evaluates **all applicable CLAP, KL, LSD, and MAE metrics**, not just CLAP. See [reported metrics and selection](../../README.md#reported-metrics-and-checkpoint-selection) for their task mapping.

From the repository root:

```bash
python inference/evaluate.py \
  --model weights/SteerSound \
  --data data/complex_val \
  --setting complex --split val \
  --output outputs/steersound_complex_reference_val
```

**Expected output:** `outputs/steersound_complex_reference_val/summary.json`. Pass that file as `--baseline-val`. This summary contains the absolute task metrics in `tasks` and the normalization values in `primary`; it is not another model weight file. If you already computed it with the same checkpoint, data, and inference settings, reuse it.

## 4. Continue training

```bash
cd STEER_post_training/emphasis/complex

torchrun --standalone --nproc_per_node=1 train.py \
  --model ../../../weights/SteerSound \
  --dit ../../../weights/SteerSound/dit/model.safetensors \
  --online-config ../../../data/my_dataset/prepared/online_data.json \
  --val ../../../data/complex_val \
  --baseline-val ../../../outputs/steersound_complex_reference_val/summary.json \
  --epochs 3 \
  --batch-size 1 \
  --workers 2 \
  --output outputs/continue_emphasis_complex
```

Validation selects the editor. After training, use the selected editor for inference and Complex test evaluation as shown below.

## 5. Use the selected editor

**Expected output:** `outputs/continue_emphasis_complex/selected.safetensors` in this training folder. Validation chooses this editor; the starting checkpoint remains eligible if later updates do not improve the score.

Return to the repository root and run:

```bash
cd ../../..

python inference/infer.py \
  --model weights/SteerSound \
  --dit STEER_post_training/emphasis/complex/outputs/continue_emphasis_complex/selected.safetensors \
  --input input.wav \
  --instruction "Remove the sound of a dog barking." \
  --output outputs/continue_emphasis_complex_demo.wav
```

**Expected result:** a mono 16 kHz, 10-second edited WAV. Use a new `--output` training directory for each run. Set process count, batch size, learning rate (`--lr`), and warmup (`--warmup`) for your training setup. Run `python train.py --help` from this training folder for options.

For the validation/test file schema and standalone evaluation command, see [evaluation data requirements](../../../inference/evaluation/README.md) and [run evaluation](../../../inference/README.md#evaluate).

## 6. Evaluate the fine-tuned editor on Complex

Run from the **repository root**, using the `complex_test` folder downloaded in step 2 and the editor selected during fine-tuning:

```bash
python inference/evaluate.py \
  --model weights/SteerSound \
  --dit STEER_post_training/emphasis/complex/outputs/continue_emphasis_complex/selected.safetensors \
  --data data/complex_test \
  --setting complex --split test \
  --output outputs/finetuned_emphasis_complex_test
```

`--dit` loads your fine-tuned editor; `--model` supplies its frozen audio/text components. The evaluator uses the ground-truth targets in `complex_test/audio.zip`.

**Expected output:** `summary.json` and `per_sample.parquet` in the specified output directory. These contain the applicable Complex metrics for each of the six tasks. Validation selects the editor; test reports its final performance.

## Directory layout

```text
complex/
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
