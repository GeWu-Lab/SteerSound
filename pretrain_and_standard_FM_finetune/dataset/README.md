# Source audio and preparation

**This workflow trains with source audio recordings and their label/caption annotations. The dataset constructs editing examples online.** You provide audio files, not prebuilt editing input/target pairs. Each step below explains its input, command, expected output, and purpose.

Choose source audio from the [datasets used in our paper](#datasets-used-in-our-paper), or use your own audio–caption data. Both choices use the same formats and preparation steps below. Obtain the paper's source audio from the original dataset pages; we provide processing tools and small format examples.

**One rule for both collections: each record refers to one complete audio file.** Duration is the full file length, and every sounding-time annotation starts from the beginning of that file. If you want an excerpt, save it as a separate audio file before annotation.

| Your starting point | Start here |
|---|---|
| You want to run the supplied examples | [Ready-to-use examples](examples/README.md#run-the-ready-to-use-examples). Their sounding times are already included. |
| You have new audio files | Step 1 below. Add text annotations first if you do not have them. |
| You have audio, text, and reviewed sounding times | Match the formats in steps 2–3, then continue at step 4. |
| You want to pretrain a new editor | Prepare data with steps 1–5, then use step 7. |

Run commands from the **repository root**, except where a `cd` is shown. Follow [Installation](../../README.md#installation), then install:

```bash
pip install -r requirements-training.txt
pip install -r requirements-metrics.txt
```

## What is constructed online?

```text
Source recordings + category labels / detailed captions + sounding times
    → AudioDataset selects recordings, applies audio transforms, and mixes sources
    → constructs an editing instruction and input/target audio for that task
    → returns a training sample (prompt_text, ref_audio, tar_audio)
```

`--online-config` selects this source-audio workflow. Source files stay on disk; instructions and editing pairs are generated in memory when samples are read. You do not need to prepare QA pairs or save edited target recordings beforehand. Source audio can contain background sounds; category-level examples should preferably have one dominant event.

## 1. Choose the source data

**Input:** your audio recordings and their text annotations. WAV and FLAC are supported. If you only have audio, add the following text manually or with an audio captioning model, then review it.

**Why:** category names identify the events to edit; detailed descriptions provide more specific target instructions.

| Collection | Text to supply | Role in training |
|---|---|---|
| **Label-level audio** | A short category such as `Dog`, `Guitar`, or `Church bell`. Prefer one dominant event per recording. | Supplies background mixtures and events to remove, replace, or modify. It is required for online mixing. |
| **Detailed-caption audio** | A sentence describing that recording, such as `A bell rings repeatedly while people talk quietly in the background.` | Supplies more descriptive Add/Generate targets and replacement destinations. This collection is optional. |

A category identifies a sound type; a caption describes a particular recording. If you only have captioned recordings, also label suitable source recordings with short categories or supply a separate label-level collection. Caption text does not isolate overlapping sound sources.

Use recordings **longer than 1 second** for a common setup across both collections. Files can have different sample rates and lengths. Metadata and sounding-time annotation cover the full file; during online construction, the loader uses the first channel, resamples to 16 kHz, and crops/pads the sampled audio to make 10-second training examples.

**Expected result:** two identified collections (or only the label-level collection), with text that matches each recording. Keep training recordings separate from validation/test recordings.

## 2. Write simple manifests

**Input:** the recordings and text from step 1.

**Why:** a manifest connects each audio file to its annotation so the tools can find and process it.

Arrange the files as follows. The two collections can contain different recordings:

```text
data/my_dataset/
├── label_level/
│   ├── manifest.csv
│   └── audio/
│       ├── dog.wav
│       ├── guitar.wav
│       └── ...
└── detailed_caption/           # omit for label-only training
    ├── manifest.csv
    └── audio/
        ├── dog.wav
        ├── guitar.wav
        └── ...
```

Write `label_level/manifest.csv` with **`path,label`**:

```csv
path,label
audio/dog.wav,Dog
audio/guitar.wav,Guitar
audio/bell.wav,Church bell
audio/rain.wav,Rain
audio/frog.wav,Frog
audio/whistling.wav,Whistling
```

Write `detailed_caption/manifest.csv` with **`path,caption`**:

```csv
path,caption
audio/dog.wav,A dog barks several times with short pauses between barks.
audio/guitar.wav,A guitar plays a slow sequence of plucked notes.
audio/bell.wav,A bell rings repeatedly while people talk quietly in the background.
```

These filenames illustrate the layout; replace them with your own. See the [supplied manifests and audio](examples/README.md) for files that can be run directly.

| Column | Required? | Format |
|---|---|---|
| `path` | Yes | Relative audio filename. `audio/dog.wav` with `--audio-root data/my_dataset/label_level` resolves to `data/my_dataset/label_level/audio/dog.wav`. |
| `label` | Label-level collection | One nonempty category string. |
| `caption` | Detailed-caption collection | One nonempty recording description. Quote a CSV field if its text contains a comma. |

The annotation tool measures the full-file `duration`; you can omit it. CSV and Parquet inputs are accepted. **Both collections use complete files.** To use part of a recording, first save that part as a new file, write its path in the manifest, and annotate times from the new file's beginning.

**Expected result:** every manifest row points to an existing file and has the appropriate text column. At this stage, **no sounding-time columns are required**. You do not need editing instructions or paired edited targets: the training loader constructs those later.

## 3. Annotate when the sound occurs

**Input:** the manifests from step 2, containing `path,label` or `path,caption`, and the corresponding audio roots.

**Why:** the loader needs sounding regions to sample short events and longer events for editing. The tool finds candidate regions by relative energy, then checks whether their sound matches the text using CLAP.

The command below performs all three operations:

| Operation | Input | Result |
|---|---|---|
| Detect candidate regions | Audio waveform. | Energy-based sounding regions lasting at least 0.5 seconds. |
| Filter with CLAP | Each candidate region and that record's `label` or `caption`. | Keep regions with text/audio similarity **≥ 0.1** by default; reject lower-scoring regions. |
| Write accepted timings | Regions that passed CLAP. | Short regions go to `point_audio`; longer regions go to `piece_audio`. |

This is **region-level filtering**. It retains the manifest row and source audio even when every candidate is rejected. In that case, the timing fields are blank. Use `--clap-threshold` to explicitly change the default threshold.

Run the label-level annotation:

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/annotate_timing.py \
  --input data/my_dataset/label_level/manifest.csv \
  --audio-root data/my_dataset/label_level \
  --kind labels --device cuda \
  --output data/my_dataset/annotated/labels.csv
```

If you also use detailed captions, run:

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/annotate_timing.py \
  --input data/my_dataset/detailed_caption/manifest.csv \
  --audio-root data/my_dataset/detailed_caption \
  --kind captions --device cuda \
  --output data/my_dataset/annotated/captions.csv
```

Use `--device cpu` for CPU processing. The default checkpoint is `laion/clap-htsat-fused`; pass `--clap-model /path/to/clap-htsat-fused` for a local copy. This is separate from the validation metric checkpoint.

**Expected files:**

```text
data/my_dataset/annotated/
├── labels.csv
└── captions.csv    # when captions are used
```

A successful command prints `Annotated N/N` and `Saved <output path>`, where `N` is the number of processed rows. Without sharding, the output retains one row per input row. Source audio files are unchanged.

**Expected columns after annotation:**

| Collection | Columns |
|---|---|
| Labels | `path,label,duration,point_audio,piece_audio` |
| Captions | `path,caption,duration,point_audio,piece_audio` |

For a 10-second label recording with accepted events at 2–2.75 s and 4–6 s, the output row could look like this. These timings illustrate the format; actual regions depend on the recording.

```csv
path,label,duration,point_audio,piece_audio
audio/dog.wav,Dog,10.0,"[(32000, 12000)]","[(64000, 32000)]"
```

A caption row retains its detailed text:

```csv
path,caption,duration,point_audio,piece_audio
audio/dog.wav,A dog barks several times with short pauses between barks.,10.0,"[(32000, 12000)]","[(64000, 32000)]"
```

| Output field | Meaning |
|---|---|
| `duration` | Full audio-file length in seconds, for either collection. |
| `point_audio` | Regions lasting **0.5 to less than 1 second**. |
| `piece_audio` | Regions lasting **at least 1 second**. |

Each timing pair is **(start sample, length in samples)** at **16 kHz**, even if the original sample rate differs. Thus `(32000, 12000)` means 2–2.75 seconds. For **both** label-level and detailed-caption records, sample 0 is the beginning of the full audio file.

A blank timing field means no candidate of that type was accepted. It does not mean that the full file is a sounding region. Review automatic proposals, particularly for overlapping sounds; energy plus CLAP does not precisely align each source in a mixture. You can supply your own reviewed timing columns in this format and skip automatic annotation.

Defaults are 50 ms energy frames, a 25 ms hop, −15 dB relative to the recording peak, and CLAP similarity ≥0.1. Candidates longer than 10 seconds are scored using their central 10 seconds.

## 4. Build the training config

**Input:** the annotated manifests from step 3, with **both timing columns present** (individual fields can be blank), plus the original audio directories.

**Why:** this converts the human-readable manifests into the loader's files and records their absolute paths. You can then use one `--online-config` argument from any training folder.

Choose **one** command according to your data.

With label-level and detailed-caption material:

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/prepare_data.py \
  --labels data/my_dataset/annotated/labels.csv \
  --label-audio-root data/my_dataset/label_level \
  --captions data/my_dataset/annotated/captions.csv \
  --caption-audio-root data/my_dataset/detailed_caption \
  --output data/my_dataset/prepared
```

With label-level material only:

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/prepare_data.py \
  --labels data/my_dataset/annotated/labels.csv \
  --label-audio-root data/my_dataset/label_level \
  --labels-only \
  --output data/my_dataset/prepared
```

With captions enabled, Add/Generate targets and replacement destinations use captioned audio. In label-only mode, those targets also come from label-level audio.

Our original training used alternative category expressions. To enable this with your collection, add `--aliases /path/to/aliases.json`; see [the dictionary format](#alternative-category-wording). Without it, training uses the original labels.

**Expected files after a successful run:**

```text
data/my_dataset/prepared/
├── label_audio.parquet
├── long_prompt.csv      # only with captions
├── label_dict.pt        # only with --aliases
└── online_data.json
```

| Output | Contents and purpose |
|---|---|
| `label_audio.parquet` | Label records with `path,duration,label,point_audio,piece_audio`. |
| `long_prompt.csv` | Detailed-caption records with `path,duration,label,point_audio,piece_audio`. The loader's legacy column name is `label`; its value is still your full caption. |
| `label_dict.pt` | The category-expression dictionary saved for the loader. |
| `online_data.json` | Absolute metadata/audio paths and online data settings. Pass **this file** to training. |

The tool automatically converts input `caption` to output `label` in the detailed-caption CSV. Category labels and caption text remain separate collections. Preparation writes metadata and a config. Editing input/target pairs are constructed online when training reads samples.

The command prints pool counts and ends with `Training config: <absolute path>/online_data.json`. It validates the data before creating the output directory; if validation fails, fix the reported issue using step 5 below. Use a new output directory when rerunning.

## 5. Check the data

**Input:** the generated `online_data.json` and the files it references.

**Why:** catch missing audio, invalid timing, and unavailable sampling pools before loading training models.

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/check_data.py \
  --config data/my_dataset/prepared/online_data.json --decode
```

**Expected terminal result:** pool counts followed by this line:

```text
PASS: files, schema, duration, 16 kHz timing bounds, and sampling pools
```

| Printed pool | What its count means |
|---|---|
| `all` | All records in the collection; should match your prepared manifest row count. |
| `point` | Records with at least one accepted short-event region. |
| `piece` | Records with at least one accepted longer-event region. |
| `point_piece` | Records with both annotation types. |

Counts refer to **records**, not the number of regions. Caption counts are `null` in label-only mode. `--decode` also reads each full audio file and checks for non-finite samples.

For the default six-task setup, provide diverse categories (at least five is a useful starting point) and both short-event and longer-event pools. Replace samples both point and piece events even when `specific_time` is `"No"`. The caption loader also requires at least one recording with both types; not every recording needs both.

If preparation or checking stops with an error:

| Reported problem | What to fix | Reason |
|---|---|---|
| Missing `label` or `caption` column | Use the right header for `--kind` during annotation, or for `--labels`/`--captions` during preparation. | The tool needs the correct text to score the sound. |
| Audio file cannot be opened | Check `path` relative to the supplied audio root. | Text and audio must refer to the same recording. |
| Duration or timing is outside the audio file | Recalculate duration and use 16 kHz **start/length** coordinates. | Out-of-bounds intervals cannot be sampled correctly. |
| A required pool is empty | Add recordings with that type of real sounding region, or use a deliberately narrower task setup. | That task cannot sample an event from an empty pool. |
| Too few distinct categories | Add categories, or explicitly reduce the background mixing/task setup in the config template. | The loader avoids using the same category for incompatible roles. |
| Output already exists | Choose a new output path. | Preparation preserves existing files. |

Do not invent short-event annotations for a continuous sound. To deliberately train fewer tasks, `prepare_data.py --task-weights` accepts six weights in this order: **Add, Remove, Generate, Replace, Volume, Pitch**. For example, `--task-weights 1 1 1 0 1 1` disables Replace. Caption mode still requires its four pools; choose `--labels-only` if you are not supplying caption data.

**Expected data during training:** the loader uses these source files to construct each editing example online:

| Field | Expected value |
|---|---|
| `prompt_text` | Generated editing instruction. |
| `ref_audio` | Input audio, shape `[1, 160000]`. |
| `tar_audio` | Target audio, shape `[1, 160000]`. |

These input/target tensors are constructed for each sampled editing task, rather than read from a stored editing-pair collection. Each tensor is mono, 16 kHz, and 10 seconds. See the [optional sample preview](examples/README.md#optional-inspect-one-generated-training-example) to inspect one. This is why source manifests do not require prewritten editing QA or pre-rendered edited targets.

## 6. Fine-tune an existing model

**Input:** the checked `online_data.json` from step 5, initialization weights, and Complex validation/test data.

**Why:** ordinary FM fine-tuning updates the editor on your source audio. Complex validation selects the checkpoint, and Complex test evaluates it afterward.

From the repository root, download weights and evaluation data:

```python
from huggingface_hub import snapshot_download

snapshot_download("GeWu-Lab/SteerSound", local_dir="weights/SteerSound")
snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["complex_val/*", "complex_test/*"],
)
```

Keep both ZIPs zipped. Prepare the [metric dependencies](../../STEER_post_training/README.md#shared-setup), then measure the initialization checkpoint on Complex validation:

```bash
python inference/evaluate.py \
  --model weights/SteerSound \
  --dit weights/SteerSound/dit/model.safetensors \
  --data data/complex_val \
  --setting complex --split val \
  --output outputs/my_dataset_initial_complex_val
```

**Expected result:** `outputs/my_dataset_initial_complex_val/summary.json`, used for validation normalization by `--baseline-val`. When using your own initialization editor, pass that same `--dit` to evaluation and fine-tuning.

```bash
cd pretrain_and_standard_FM_finetune

torchrun --standalone --nproc_per_node=1 finetune.py \
  --model ../weights/SteerSound \
  --dit ../weights/SteerSound/dit/model.safetensors \
  --online-config ../data/my_dataset/prepared/online_data.json \
  --val ../data/complex_val \
  --baseline-val ../outputs/my_dataset_initial_complex_val/summary.json \
  --epochs 3 --batch-size 1 --workers 2 \
  --output outputs/my_dataset
```

This runs standard Flow Matching on editing examples constructed online from your prepared audio–caption data. Complex validation selects `outputs/my_dataset/selected.safetensors` in this training folder. For Emphasis or Recovery, use the separate [STEER guides](../../STEER_post_training/README.md).

Return to the repository root and evaluate the selected editor on Complex test:

```bash
cd ..

python inference/evaluate.py \
  --model weights/SteerSound \
  --dit pretrain_and_standard_FM_finetune/outputs/my_dataset/selected.safetensors \
  --data data/complex_test \
  --setting complex --split test \
  --output outputs/my_dataset_complex_test
```

**Expected result:** `summary.json` and `per_sample.parquet` in the evaluation output directory. For validation evaluation and inference examples, see the [training guide](../README.md#4-evaluate-on-complex).

## 7. Or pretrain a new editor

**Input:** the same checked config and downloaded frozen components.

**Why:** this starts a new DiT instead of adapting an existing editor. Data formatting and online construction are the same.

From the repository root:

```bash
cd pretrain_and_standard_FM_finetune

torchrun --standalone --nproc_per_node=1 train.py \
  --model ../weights/SteerSound \
  --online-config ../data/my_dataset/prepared/online_data.json \
  --epochs 10 --save-every 5 \
  --output outputs/my_pretrain
```

After training, evaluate an exported epoch checkpoint, for example `pretrain_and_standard_FM_finetune/outputs/my_pretrain/dit_epoch010.safetensors`, on Complex val/test using the [evaluation commands](../README.md#4-evaluate-on-complex). Pretraining runs without in-loop validation; compare epoch checkpoints on validation before testing the chosen checkpoint.

## Alternative category wording

For label-level audio, we use audio recordings selected from [Hive](https://github.com/AlayaLab/Hive) together with their corresponding Hive-provided labels. Label replacement augments those labels with alternative category expressions: the loader varies the short category phrase used in training instructions. For example, an `aliases.json` file can contain:

```json
{"Guitar": ["guitar music", "guitar sound", "guitar playing"]}
```

The loader randomly chooses `Guitar` or one of these phrases when constructing an instruction. This mechanism was enabled in our original training. For your own collection, provide phrases matching its category labels, or omit `--aliases` to use the original labels. The dictionary is separate from the audio manifests and detailed captions.

Dictionary keys must exactly match category strings in your label manifest. You can provide alternatives for only a subset of categories; the remaining categories keep their original wording.

## Datasets used in our paper

The Complex audio editing dataset used in our paper is constructed online from the following audio sources. The table shows which sources are used in the label-level and detailed-caption training pools:

- **Label-level pool:** short sound categories identify background sounds and events to edit.
- **Detailed-caption pool:** recording-specific descriptions provide targets for Add/Generate and replacement destinations.

| Original dataset / official link | Label-level pool | Detailed-caption pool |
|---|---|---|
| [🔗 BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/) | ✓ | — |
| [🔗 AudioSet](https://research.google.com/audioset/) | ✓ | — |
| [🔗 VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) | ✓ | ✓ |
| [🔗 AVSpeech](https://looking-to-listen.github.io/avspeech/) | ✓ | — |
| [🔗 MUSIC / MUSIC21](https://github.com/roudimit/MUSIC_dataset) | ✓ | — |
| [🔗 Clotho v2](https://zenodo.org/records/4783391) | ✓ | ✓ |
| [🔗 SoundBible](https://soundbible.com/) | ✓ | — |
| [🔗 AVE](https://github.com/YapengTian/AVE-ECCV18) | ✓ | — |
| [🔗 VoiceBank-DEMAND](https://doi.org/10.7488/ds/2117) | ✓ | — |
| [🔗 DCASE 2024 Task 9](https://dcase.community/challenge2024/task-language-queried-audio-source-separation) | ✓ | — |
| [🔗 ESC-50](https://github.com/karolpiczak/ESC-50) | ✓ | — |
| [🔗 Freesound](https://freesound.org/) | ✓ | — |
| [🔗 FSD50K](https://zenodo.org/records/4060432) | ✓ | ✓ |
| [🔗 AudioCaps](https://github.com/cdjkim/audiocaps) | — | ✓ |
| [🔗 MACS](https://zenodo.org/records/5114771) | — | ✓ |

For label-level audio, we use audio recordings selected from [Hive](https://github.com/AlayaLab/Hive) together with their corresponding Hive-provided labels, and augment those labels through [label replacement with alternative category expressions](#alternative-category-wording).

For MACS, obtain the [caption annotations](https://zenodo.org/records/5114771) and [TAU Urban Acoustic Scenes 2019 source audio](https://zenodo.org/records/2589280) separately. Use the original dataset pages for current access instructions.

These datasets are the source audio for online construction. Follow steps 1–5 above to prepare their audio and label/caption annotations, or use your own recordings in the same format.

## Files

| Path | Purpose |
|---|---|
| [examples/](examples/README.md) | Nine audio examples and manifests for both source types. |
| [preparation/](preparation/README.md) | Sounding-time annotation, conversion, and preflight tools. |
| `dataset.py` | Read source recordings and construct editing tasks online. |
| `paired.py` | Read paired Complex validation/test records and their audio from ZIP. |
| `balanced_sampler.py` | Balanced six-task sampling for paired data. |
| `worker_determinism.py` | Reproducible per-epoch and per-worker data sampling. |
| `create_example_data.py` | Optional synthetic-tone format generator. |
