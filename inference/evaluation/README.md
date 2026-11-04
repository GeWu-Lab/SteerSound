# Evaluation data requirements

The evaluator reads an evaluation collection, generates one edit per record, and reports the metrics for the chosen setting. Training-source manifests (`path,label` / `path,caption`) are inputs to online training; evaluation uses the record format below.

## Files and audio format

```text
data/<evaluation_split>/
├── metadata.parquet
├── audio.zip
└── manifest.json          # included in the public release
```

`metadata.parquet` describes the tasks and points to WAV members inside `audio.zip`. Keep the ZIP zipped. Custom evaluation collections need the metadata and ZIP; the public manifest records integrity information.

Every scored WAV must be **mono, 16 kHz, 10 seconds** (160,000 samples), with finite values. ZIP paths are relative member names, such as `audios/test/remove_000001_input.wav`. Evaluation must cover all six tasks. Record IDs must be unique, and every row's `split` must match the command's `--split val` or `--split test`.

## Selective data

Run from the repository root. Download validation/test data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["selective_val/*", "selective_test/*"],
)
```

**Expected result:** `data/selective_val` and `data/selective_test`, each containing `metadata.parquet`, `audio.zip`, and `manifest.json`. Each split has 6,000 records: 1,000 for each task.

Selective scoring uses the input recording, instruction, and task annotations. Reference-target audio is not required by these success/error rules. The public tables already include the required annotations, so use them unchanged for the released benchmark.

| Common metadata field | Meaning |
|---|---|
| `sample_id` | Unique example ID. |
| `split` | `val` or `test`. |
| `task`, `task_id` | `add=0`, `remove=1`, `generate=2`, `replace=3`, `volume=4`, `pitch=5`. |
| `instruction` | Plain editing instruction. |
| `input_audio_path` | Input WAV member in the ZIP. Generate uses silence instead of reading an input WAV. |
| `target_audio_path` | Column retained in the schema; may be empty for Selective evaluation. |
| `change_vol` | Requested dB change, or `0.0` for other tasks. |
| `change_Pitch` | Requested semitone change, or `0.0` for other tasks. |

Task-specific Selective annotations:

| Tasks | Fields used for scoring | Meaning |
|---|---|---|
| Generate, Add | `target_label` | Text describing the requested sound. |
| Replace | `source_label`, `destination_label`, `base_labels` | Original sound, replacement sound, and background labels. |
| Replace, Remove, Pitch, Volume | `edit_start_sample`, `edit_end_sample`, `preserve_start_sample`, `preserve_end_sample` | Edit and preserve regions, expressed as **start/end** samples at 16 kHz. |
| Pitch, Volume | `change_Pitch` or `change_vol` | Requested change used by the task-success rule. |

These region fields are start/end coordinates. The sounding-time lists used in source-audio preparation are start/length pairs; each schema's definitions are given in its own guide.

## Complex data

Run from the repository root. Download validation/test data from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "GeWu-Lab/SteerSound", repo_type="dataset", local_dir="data",
    allow_patterns=["complex_val/*", "complex_test/*"],
)
```

**Expected result:** `data/complex_val` and `data/complex_test`, each with `metadata.parquet`, `audio.zip`, and `manifest.json`.

Both splits include input and ground-truth target WAVs, including targets for Generate. Every `target_audio_path` names a target WAV in `audio.zip`. The downloaded data is ready for full Complex evaluation.

The same layout and fields apply to your own paired evaluation collection:

```text
data/
├── complex_val/
│   ├── metadata.parquet
│   ├── audio.zip          # input and ground-truth target WAVs
│   └── manifest.json
└── complex_test/           # same three files
```

The common metadata fields listed above also apply to Complex. In addition:

| Field | Meaning |
|---|---|
| `target_audio_path` | Required nonempty ZIP member name for the reference target of every record. |
| `base_labels` | Background sound labels, stored as a list. |
| `add_labels` | Added/generated/replacement sound labels or descriptions, or `[]`. |
| `remove_labels` | Removed/replaced sound labels, or `[]`. |
| `start_time`, `end_time` | Optional edit boundaries in seconds; use `0.0, 10.0` for full-duration edits. |

For example, one `metadata.parquet` record shown as JSON:

```json
{
  "sample_id": "remove_000001",
  "split": "test",
  "task": "remove",
  "task_id": 1,
  "instruction": "Remove the sound of a dog barking.",
  "input_audio_path": "audios/test/remove_000001_input.wav",
  "target_audio_path": "audios/test/remove_000001_target.wav",
  "change_vol": 0.0,
  "change_Pitch": 0.0,
  "base_labels": ["rain"],
  "add_labels": [],
  "remove_labels": ["dog barking"],
  "start_time": 0.0,
  "end_time": 10.0
}
```

Here the input contains rain and barking; the reference target contains the rain after removal. This is a format example. Each released split contains all six tasks and their corresponding ground-truth targets.

## Run evaluation

Follow [the evaluation commands](../README.md#evaluate) after preparing the data and metric dependencies. `--data` identifies the folder containing `metadata.parquet` and `audio.zip`; `--split` must match that table.

**Expected outputs:**

| File | Contents |
|---|---|
| `summary.json` | Complex task metrics, or Selective task success rates and overall success/error rates, with valid/excluded counts. |
| `per_sample.parquet` | One row per record with its metrics and, for Selective, task-success/error classification. |

An optional Complex `--baseline` summary adds `TBG_all` / `TBG_sel`. Absolute metrics work without a baseline. Missing Complex references are reported as an error rather than omitted silently.
