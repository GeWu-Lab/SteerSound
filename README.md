# Steer Unified Audio Editing Toward Reliable Source Selection

Instruction-based audio editing with STEER-Emphasis and STEER-Recovery for our base editor.

- [Model: GeWu-Lab/SteerSound](https://huggingface.co/GeWu-Lab/SteerSound)
- [Data: GeWu-Lab/SteerSound](https://huggingface.co/datasets/GeWu-Lab/SteerSound)

## Model components

The audio VAE and DiT editor are trained by the SteerSound team. The text encoder/tokenizer and vocoder use pretrained weights from the sources below.

| Component | Source | Hugging Face weight directory |
|---|---|---|
| Audio VAE (encoder and decoder) | Trained by us | `vae/` |
| DiT audio editor | Trained by us | `dit/` |
| T5 text encoder and tokenizer | [google-t5/t5-base](https://huggingface.co/google-t5/t5-base), using its encoder | `text_encoder/` |
| HiFi-GAN vocoder | [cvssp/audioldm2 — vocoder](https://huggingface.co/cvssp/audioldm2/tree/main/vocoder) | `vocoder/` |

## Install for NVIDIA GPUs

Use Linux, Python 3.10 or 3.11, an NVIDIA GPU and a compatible NVIDIA driver. The following is a pinned CUDA 12.6 installation target; CUDA execution has not been tested on the release preparation host.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e . --no-deps
```

PyTorch supplies its CUDA runtime in these wheels. Training additionally needs a suitable CUDA toolkit/compiler for DeepSpeed extensions: `pip install -r requirements-training.txt`. Metrics additionally need `pip install -r requirements-metrics.txt`. Native PyTorch SDPA is used; installing FlashAttention is optional and unnecessary for the default configuration. See [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/) and [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/).

## Inference

```bash
python inference/infer.py --model GeWu-Lab/SteerSound   --instruction "Remove the sound of a dog barking."   --input input.wav --output edited.wav

python inference/infer.py --model GeWu-Lab/SteerSound   --instruction "Generate the sound of rain." --output rain.wav
```

```python
import soundfile as sf
from steersound import SteerSoundPipeline

pipe = SteerSoundPipeline.from_pretrained("GeWu-Lab/SteerSound", device="cuda")
wave = pipe("Increase the volume of the sound of a dog barking by 3.0 dB.", audio="input.wav")
sf.write("edited.wav", wave, pipe.sample_rate, subtype="PCM_16")
```

Users supply a plain instruction and an input waveform for editing. Generation needs no input audio. The API infers the task and amount automatically; it does not require task IDs or model-control tokens. File audio is downmixed to mono, resampled to 16 kHz, truncated or right-padded to 10 seconds. Array inputs require `sample_rate=...` and mono or channels×samples layout. The output is a 16 kHz mono 10-second waveform. A local model directory may replace the Hub model ID.

The model owns its control-token prefix and inserts it in the text encoder. Empty classifier-free guidance conditioning remains empty. Public dataset instructions contain no such prefix. Evaluation uses the exact numeric metadata rather than recovering a rounded value from the instruction.

### All six tasks

| Task | Plain instruction | Numeric conditioning |
|---|---|---|
| Generation | `Generate the sound of rain.` | None |
| Addition | `Add the sound of a dog barking.` | None |
| Removal | `Remove the sound of a dog barking.` | None |
| Replacement | `Replace the sound of a dog barking by the sound of a cat meowing.` | None |
| Volume | `Increase the volume of the sound of a dog barking by 3.0 dB.` | +3 dB |
| Pitch | `Reduce the pitch of the sound of a dog barking by 2.0 semitones.` | −2 semitones |

Use `Increase` or `Reduce` and an explicit nonnegative magnitude for pitch/volume. Volume supports −5 to +5 dB; pitch supports −4 to +4 semitones in steps of 0.5. Native dataset instructions omit units; units in user input are converted internally to the native one-decimal, unit-free text template. For temporal editing, place the region before the amount, e.g. `Increase the volume of the sound of a dog barking from 2.0 to 4.0 seconds by 3.0 dB.` Use one operation per instruction. This deterministic parser supports the documented English templates; it does not use another language model or guess missing amounts. Unsupported or ambiguous instructions raise an error. Time intervals are conveyed through text to the editor, rather than hard masks applied to the output.

## Model size and GPU memory

| Component | Parameters |
|---|---:|
| DiT editor | 0.673B |
| Audio VAE | 0.509B |
| T5 encoder | 0.110B |
| HiFi-GAN vocoder | 0.055B |
| Total | **1.348B** |

**GPU memory for inference: approximately 8 GB** (BF16, batch size 1, 10-second audio).

## Training

Each subprogram contains independent model, objective, data and metric implementations. Run its `train.py` from its own folder. All fine-tuning updates every DiT parameter.

| Entry point | Initialization | Epochs | Peak LR | Warmup updates | Schedule | Validation |
|---|---|---:|---:|---:|---|---|
| `pretrain` | Random DiT; supplied frozen audio/text components | 70 | 1e-4 | 1000 | Cosine | No selection |
| `emphasis/complex` | Base editor | 20 | 2e-6 | 1000 | Cosine | Every 0.5 epoch, TBG_all |
| `emphasis/selective` | Base editor | 20 | 2e-6 | 1000 | Cosine | Every epoch, overall success |
| `recovery/complex` | Selected Emphasis | 2 per k | 1e-5 | 64 per k | Constant after warmup | Every 0.5 epoch, TBG_all |
| `recovery/selective` | Selected Emphasis | 5 per k | 1e-5 | 64 per k | Constant after warmup | Every epoch, overall success |

Fine-tuning defaults to microbatch 30. With 16 GPUs and accumulation 1 the global batch is 480. For example 4 GPUs with microbatch 30 and accumulation 4 have the same global batch, though performance/memory have not been benchmarked. Defaults can be explicitly overridden for available hardware. Pretraining defaults to microbatch 1; its archived single-host configuration had global batch 16.

Emphasis samples Add/Generate from Logit-Normal(0,1), Remove uniformly from [0.97,1], and Replace/Volume/Pitch uniformly from [0.71,1]. Pretraining uses standard FM with Logit-Normal(0,1) for all tasks. Pretraining/Emphasis use the original DeepSpeed WarmupCosineLR (its default logarithmic warmup); Recovery uses linear warmup with applied update 1 = peak/64 and update 64 = peak. Selective Recovery preserves the original balanced sampler: every microbatch has equal counts of all six tasks, so its microbatch must be divisible by six. Recovery retains standard FM for Add/Generate and uses detached k-step Euler rollout with corrective target `(z_k-z_target)/sigma_k` for the four editing tasks. Each k starts from the best validation checkpoint of the preceding stage (including its E0 candidate), with a fresh AdamW optimizer and scheduler. Ties retain the input or earliest selected checkpoint. Both Recovery entry points default to patience 2: stop after two consecutive depths fail to improve the global validation best, with a maximum depth of 15. The test split is evaluated once after selection, if supplied.

Complete online pretraining and Complex training require the original raw audio corpus, timing parquet, long-prompt CSV/audio and label dictionary. These raw training assets are **not included** in this data release. `online_data.example.json` preserves the original augmentation settings and makes those paths explicit. Copy it, set paths to your own licensed/source assets, and pass `--online-config`. The released `selective_train` is a static paired editing set; it is a separate training option and does not reproduce full online corpus pretraining.

The released final Recovery model provides frozen components for all entry points. For Emphasis supply your separately pretrained base DiT with `--dit`. For Recovery supply the validation-selected Emphasis DiT. The final Recovery DiT should not be used as a substitute base initialization when reproducing the reported stages.

See the README inside each training folder for complete commands and the required baseline summaries. Checkpoints include DeepSpeed training state and selected inference-only `.safetensors`; this first portable runner starts fresh and does not offer mid-epoch resume.

## Data and evaluation

Download the released audio archives without unpacking:

```python
from huggingface_hub import snapshot_download
snapshot_download("GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
                  allow_patterns=["complex_val/*", "complex_test/*", "selective_train/*", "selective_val/*", "selective_test/*", "README.md"])
```

Each folder contains `metadata.parquet`, `audio.zip` and a SHA256 manifest. Readers access WAV members directly from ZIP. Internal paths preserve the original `audios/<split>/...` layout, even though the public top folders have composite names. All validation/test splits (Complex and Selective) have input audio only; empty target paths are intentional. Only selective_train contains paired target audio. Remaining input WAV bytes and metadata values are preserved, apart from removing model prefixes and clearing evaluation target paths.

```bash
python inference/evaluate.py --model GeWu-Lab/SteerSound   --data /path/to/private_complex_val --setting complex --split val --output outputs/base_val
# Add --dit base_editor.safetensors to evaluate a specific base DiT.
python inference/evaluate.py --model GeWu-Lab/SteerSound   --data data/selective_test --setting selective --split test --output outputs/selective_test
```

The published Complex val/test splits are intended for inference and do not include ground-truth targets. The full Complex metric suite (KL, pitch/volume MAE, LSD and the resulting TBG_all) requires privately held paired targets, using the same metadata/archive layout. The Complex evaluation command above and Complex training validation therefore require your private paired validation/test data. The evaluator rejects missing references explicitly rather than reporting fabricated or partial scores. Selective success evaluation works with the public input-only splits.

Before using CLAP metrics, download the original [LAION larger CLAP](https://huggingface.co/laion/larger_clap_general):

```python
from huggingface_hub import snapshot_download
from pathlib import Path
snapshot_download("laion/larger_clap_general", local_dir=Path.home()/".cache/steersound/clap")
```

Set `STEERSOUND_CLAP_PATH` to another local CLAP directory if needed. PaSST weights for softmax KL download through the retained PaSST model factory into the standard Torch cache (`TORCH_HOME` may override it). Run metrics on CPU by default, or set `--metric-device cuda` when enough VRAM is available.

Complex reports only task-aware CLAP, softmax KL, pitch/volume MAE, LSD and TBG. CLAP is the primary score for Add/Generate/Remove/Replace, and MAE for Pitch/Volume. `TBG_all=100*(geometric_mean(task_ratios)-1)` uses equal task weights; ratios invert for MAE. `TBG_sel` uses Remove/Replace/Pitch/Volume. TBG requires a separate base-model summary for the same split; selection uses only the validation baseline. A summary without a baseline reports absolute metrics and `primary`, which can then be used as `--baseline` or `--baseline-val`. Complex temporal metrics follow edit/complement regions; the released complex split edits the full duration.

Selective evaluation retains the original frozen task-success/error rules. Pitch pairs that fail measurability gates are excluded from both success numerator and denominator. Summaries pool raw per-sample counts, rather than average per-rank rates. `metrics/` contains the standalone implementation; experiment-local copies avoid dependencies on sibling programs.

## Validation status

Release checks include source/data integrity, instruction parsing and prefix-tokenizer equivalence, strict checkpoint loading, CPU scientific-objective tests and CPU inference component checks. This host has no NVIDIA GPU: CUDA training, GPU throughput and peak VRAM have not been measured. The portable GPU runners preserve the scientific objectives and selection rules, while adapting orchestration and paths for publication; they are not a claim that the archived training runs were executed with this CUDA environment.

CPU protocol checks: `PYTHONPATH=inference python -m unittest discover -s tests -v`.
