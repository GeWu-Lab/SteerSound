<div align="center">

# Steer Unified Audio Editing Toward Reliable Source Selection

[![Project Page](https://img.shields.io/badge/Project-Page-2563EB?style=flat-square&logo=googlechrome&logoColor=white)](https://gewu-lab.github.io/SteerSound/) [![GitHub Code](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/GeWu-Lab/SteerSound) [![Hugging Face Weights](https://img.shields.io/badge/Hugging_Face-Weights-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/GeWu-Lab/SteerSound) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-FF9D00?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/GeWu-Lab/SteerSound)

</div>

SteerSound supports instruction-based audio generation and editing, with STEER-Emphasis and STEER-Recovery for post-training.

## Installation

Use Python 3.10 or 3.11.

```bash
git clone https://github.com/GeWu-Lab/SteerSound.git
cd SteerSound

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e . --no-deps
```

## Inference

Edit an audio file:

```bash
python inference/infer.py \
  --model GeWu-Lab/SteerSound \
  --input input.wav \
  --instruction "Remove the sound of a dog barking." \
  --output edited.wav
```

Generate audio:

```bash
python inference/infer.py \
  --model GeWu-Lab/SteerSound \
  --instruction "Generate the sound of rain." \
  --output rain.wav
```

Or use the Python API:

```python
import soundfile as sf
from steersound import SteerSoundPipeline

pipe = SteerSoundPipeline.from_pretrained("GeWu-Lab/SteerSound", device="cuda")
wave = pipe(
    "Increase the volume of the sound of a dog barking by 3.0 dB.",
    audio="input.wav",
)
sf.write("edited.wav", wave, pipe.sample_rate, subtype="PCM_16")
```

| Task | Example instruction |
|---|---|
| Generate | `Generate the sound of rain.` |
| Add | `Add the sound of a dog barking.` |
| Remove | `Remove the sound of a dog barking.` |
| Replace | `Replace the sound of a dog barking by the sound of a cat meowing.` |
| Volume | `Increase the volume of the sound of a dog barking by 3.0 dB.` |
| Pitch | `Reduce the pitch of the sound of a dog barking by 2.0 semitones.` |

Output audio is mono, 16 kHz, and 10 seconds long. Use one operation per instruction; volume changes support ±5 dB and pitch changes support ±4 semitones in steps of 0.5.

## Training

### Pretraining and fine-tuning with standard FM

Use [pretrain_and_standard_FM_finetune/](pretrain_and_standard_FM_finetune/README.md) to train with the standard Flow Matching objective:

| Task | Entry point | Initialization |
|---|---|---|
| Pretrain a new editor | `pretrain_and_standard_FM_finetune/train.py` | New DiT; use the downloaded frozen audio/text components. |
| Fine-tune on your own data | `pretrain_and_standard_FM_finetune/finetune.py` | Load an existing editor and update it with standard Flow Matching. |

Follow [Source audio and preparation](pretrain_and_standard_FM_finetune/dataset/README.md) to prepare the paper's source audio or your own audio–caption data, then [pretrain or fine-tune](pretrain_and_standard_FM_finetune/README.md). Editing examples are constructed online; both modes use Complex val/test for evaluation.

### STEER post-training

Fine-tune the released SteerSound checkpoint for downstream audio editing using Emphasis or Recovery. Choose a workflow and follow the [STEER guides](STEER_post_training/README.md):

| Workflow | Training data | Emphasis | Recovery |
|---|---|---|---|
| Complex audio editing data | Prepare the paper's source audio or your own audio–caption data using [Source audio and preparation](pretrain_and_standard_FM_finetune/dataset/README.md). Construct editing examples online; evaluate on Complex val/test. | [Guide](STEER_post_training/emphasis/complex/README.md) | [Guide](STEER_post_training/recovery/complex/README.md) |
| Selective audio editing data | Editing pairs from our released [🤗 Selective audio editing data](https://huggingface.co/datasets/GeWu-Lab/SteerSound). | [Guide](STEER_post_training/emphasis/selective/README.md) | [Guide](STEER_post_training/recovery/selective/README.md) |

See [available Hugging Face assets and workflow choices](STEER_post_training/README.md#what-is-available-on-hugging-face) for the download layout and the roles of the released splits.

Each guide lists the data and weights, download/preparation steps, training command, and selected editor to use for inference. Standard fine-tuning in `pretrain_and_standard_FM_finetune/` and STEER post-training are separate training routes.

## Evaluation

Run these commands from the **repository root**. Both settings use `metadata.parquet` and mono 16 kHz, 10-second WAV members inside `audio.zip`, covering all six tasks. Keep ZIPs zipped. See [metadata fields and audio requirements](inference/evaluation/README.md).

Install metric dependencies and prepare CLAP:

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

KL also uses PaSST, whose weights download automatically on first use. LSD and Pitch/Volume MAE need no pretrained weights. See [metric dependencies](metrics/README.md#metric-dependencies).

### Selective evaluation

Download validation/test data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["selective_val/*", "selective_test/*"],
)
```

**Expected folders:** `data/selective_val` and `data/selective_test`, each with `metadata.parquet`, `audio.zip`, and `manifest.json`. The downloaded input audio and annotations are ready for Selective success/error evaluation; its rules do not require reference-target audio.

```bash
python inference/evaluate.py \
  --model GeWu-Lab/SteerSound \
  --data data/selective_test \
  --setting selective --split test \
  --output outputs/selective_test
```

### Complex evaluation

Download validation/test data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["complex_val/*", "complex_test/*"],
)
```

**Expected folders:** `data/complex_val` and `data/complex_test`, each with `metadata.parquet`, `audio.zip`, and `manifest.json`.

The downloaded Complex splits include input audio, instructions, and ground-truth target audio for full evaluation. Run:

```bash
python inference/evaluate.py \
  --model GeWu-Lab/SteerSound \
  --data data/complex_test \
  --setting complex --split test \
  --output outputs/complex_test
```

For validation, use `--split val` with the corresponding validation folder. **Expected outputs:** `summary.json` and `per_sample.parquet`. Add `--dit /path/to/editor.safetensors` to evaluate an editor you trained. Complex `--baseline /path/to/reference_summary.json` optionally adds TBG summaries. Use a new output directory per run. See the [evaluation walkthrough](inference/README.md#evaluate).

## Model Components

| Component | Parameters | Source |
|---|---:|---|
| DiT editor | 0.673B | SteerSound |
| Audio VAE | 0.509B | SteerSound |
| T5 encoder | 0.110B | [T5-base](https://huggingface.co/google-t5/t5-base) |
| HiFi-GAN vocoder | 0.055B | [AudioLDM2 vocoder](https://huggingface.co/cvssp/audioldm2/tree/main/vocoder) |
| **Total** | **1.348B** | |

Approximate inference memory: **8 GB** with BF16, batch size 1, and 10-second audio.

## Acknowledgements

We thank the teams behind the following projects for their open-source contributions:

- [AudioLDM2](https://github.com/haoheliu/AudioLDM2) for the pretrained HiFi-GAN vocoder used for audio decoding.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer) for the pretrained text encoder and tokenizer.
- [Hive](https://github.com/AlayaLab/Hive) for the audio recordings and corresponding labels selected for our label-level audio pool.
- [MMAE](https://github.com/ddlBoJack/MMAE) for the audio samples used in our demos.
