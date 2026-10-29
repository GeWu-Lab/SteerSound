# Audio and manifest examples

This page shows the two source formats and lets you check data loading with nine supplied recordings. Both collections use complete audio files, and all timings start at the beginning of the referenced file. The dataset constructs editing examples online from these source recordings. For training on your own collection, follow [Train with your own audio](../README.md).

| Included file | What it contains | When to use it |
|---|---|---|
| `audio/*.flac` | Real recordings converted to mono 16 kHz. | Referenced by the manifests. |
| `manifest.csv` | Audio paths and category labels or detailed captions. | Input to sounding-time annotation. |
| `annotated.csv` | The same records with existing automatic sounding-time annotations. | Use this directly for the quick start below. |
| `label_level/aliases.json` | Alternative short expressions for the category labels. | Passed to `--aliases` in the quick start. Our original training used this augmentation. |

## Shared format

| Stage | Label-level records | Detailed-caption records |
|---|---|---|
| Before time annotation | `path,label` | `path,caption` |
| After time annotation | `path,label,duration,point_audio,piece_audio` | `path,caption,duration,point_audio,piece_audio` |

The only difference is the text column: `label` names the event category; `caption` describes the recording. `path` identifies a full audio file. `duration` is its length in seconds. `point_audio` and `piece_audio` list short and longer sounding regions as **(start sample, length in samples)** at 16 kHz, measured from the full file's beginning.

These are source-audio manifests. They do not contain an editing instruction or an input/target editing pair; `AudioDataset` creates those when a training sample is requested.

## Label-level examples

See [input manifest](label_level/manifest.csv), [annotated manifest](label_level/annotated.csv), and optional [alternative category wording](label_level/aliases.json). These AudioSet examples come from audio recordings selected from Hive and use their corresponding Hive-provided labels.

| Audio | Category (`label`) | Original AudioSet ID |
|---|---|---|
| [🔊 Guitar](label_level/audio/00_HN179ttKxBg.flac) | `Guitar` | `HN179ttKxBg` |
| [🔊 Whistling](label_level/audio/01_Wi3K4j5yUpU.flac) | `Whistling` | `Wi3K4j5yUpU` |
| [🔊 Bird](label_level/audio/02_2kUXFScf1Es.flac) | `Bird` | `2kUXFScf1Es` |
| [🔊 Bell](label_level/audio/03_KMtlxzHkHU8.flac) | `Church bell` | `KMtlxzHkHU8` |
| [🔊 Pouring](label_level/audio/04_JEeZR7JAbtY.flac) | `Pour` | `JEeZR7JAbtY` |
| [🔊 Frog](label_level/audio/05_JL4e3gwQERw.flac) | `Frog` | `JL4e3gwQERw` |

## Detailed-caption examples

See [input manifest](detailed_caption/manifest.csv) and [annotated manifest](detailed_caption/annotated.csv). These examples come from the Clotho development split. Their captions describe particular recordings, rather than naming a category.

| Audio | Detailed description (`caption`) |
|---|---|
| [🔊 Pouring water](detailed_caption/audio/00_RKeaton_EMF366_8_Running_Water_in_Small_Porcelain_Sink.flac) | A person pours liquid from a container into another body of water |
| [🔊 Bell with background voices](detailed_caption/audio/01_20110220_churchbell.flac) | A bell is ringing loudly again and again while people are talking in the background. |
| [🔊 Conversations](detailed_caption/audio/02_Greek_Chat2.flac) | A large group of people are having several conversations in the background. |

The bell examples make the distinction explicit: the category is `Church bell`, while a caption also describes repetition and background voices. The optional category-wording dictionary only lists different names for the same category.

Sources: [AudioSet](https://research.google.com/audioset/), [Hive](https://github.com/AlayaLab/Hive), and [Clotho](https://zenodo.org/records/4783391). Original Clotho filenames: `RKeaton_EMF366_8_Running Water in Small Porcelain Sink.wav`, `20110220_churchbell.wav`, and `Greek Chat2 - (Apollonia&#39;s sPA) 18_44 05.10.wav`. These are small format demonstrations, not a training/validation benchmark.

## Run the ready-to-use examples

The supplied `annotated.csv` files already contain sounding times. Start with those files. Run both commands from the **repository root**, after installing the [training dependencies](../README.md).

### 1. Prepare the config

**Input:** six label-level records, three detailed-caption records, their audio, and the category-wording dictionary. The included `annotated.csv` files already have accepted sounding regions; the command uses those annotations directly.

**Why:** create the loader's metadata files and one config containing their absolute paths.

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/prepare_data.py \
  --labels pretrain_and_standard_FM_finetune/dataset/examples/label_level/annotated.csv \
  --label-audio-root pretrain_and_standard_FM_finetune/dataset/examples/label_level \
  --captions pretrain_and_standard_FM_finetune/dataset/examples/detailed_caption/annotated.csv \
  --caption-audio-root pretrain_and_standard_FM_finetune/dataset/examples/detailed_caption \
  --aliases pretrain_and_standard_FM_finetune/dataset/examples/label_level/aliases.json \
  --output data/example_sources/prepared
```

**Expected output:**

```text
data/example_sources/prepared/
├── label_audio.parquet   # 6 label-level records
├── long_prompt.csv      # 3 detailed-caption records
├── label_dict.pt        # supplied alternative category expressions
└── online_data.json     # config to pass to the loader or training
```

The terminal ends with `Training config: <absolute path>/online_data.json`. Audio files remain in the supplied example folders. Editing input/target pairs are constructed later by the loader.

### 2. Check the prepared data

**Input:** the config and files from step 1.

**Why:** confirm that audio can be read, timings fit the recordings, and the required event pools exist.

```bash
python pretrain_and_standard_FM_finetune/dataset/preparation/check_data.py \
  --config data/example_sources/prepared/online_data.json --decode
```

**Expected output for these supplied annotations:**

| Collection | `all` | `point` | `piece` | `point_piece` |
|---|---:|---:|---:|---:|
| Labels | 6 | 6 | 6 | 6 |
| Captions | 3 | 3 | 3 | 3 |

These are counts of records, not individual sounding regions. The final line is:

```text
PASS: files, schema, duration, 16 kHz timing bounds, and sampling pools
```

The example data is now ready for the loader. To fine-tune, prepare your actual collection using the [new-data workflow](../README.md), then run its [fine-tuning command](../README.md#6-fine-tune-an-existing-model). These nine recordings are intended to demonstrate the format and data loading.

## Optional: inspect one generated training example

Skip this section if the data check is enough. This preview shows what `AudioDataset` constructs from the source recordings:

| Sample field | Meaning |
|---|---|
| `prompt_text` | The editing instruction generated for this sample. |
| `ref_audio` | Input audio before editing. |
| `tar_audio` | Target audio after editing. |

Run this block from the **repository root**. It temporarily enters `pretrain_and_standard_FM_finetune/`, reads one sample, and returns to the root. It only loads data; it does not start training or load the editing model.

```bash
cd pretrain_and_standard_FM_finetune

python - <<'PY'
import json
from pathlib import Path
from dataset.dataset import AudioDataset

config = json.loads(Path("../data/example_sources/prepared/online_data.json").read_text())
# Skip pitch-transform warmup when previewing one sample.
config["pitch_shift_prewarm"] = False
dataset = AudioDataset(**config)
sample = dataset[0]
print("Instruction:", sample["prompt_text"])
print("Input audio:", sample["ref_audio"].shape)
print("Target audio:", sample["tar_audio"].shape)
PY
cd ..
```

Both audio shapes should be `torch.Size([1, 160000])`: one channel and 160,000 samples, or 10 seconds at 16 kHz. The instruction and selected recordings vary between samples.

## For your own recordings

Start with `path,label` and `path,caption` manifests, then follow [step 3: annotate when the sound occurs](../README.md#3-annotate-when-the-sound-occurs). That command detects candidates, filters them with CLAP, and adds duration/timing columns. Use its annotated outputs to prepare and check your training config, then fine-tune. If you already have reviewed sounding-time annotations, start at [step 4: build the training config](../README.md#4-build-the-training-config).
