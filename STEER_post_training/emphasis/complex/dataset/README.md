# Dataset loaders

| File | Purpose |
|---|---|
| `dataset.py` | Construct editing instructions and input/target audio online from labeled source recordings and detailed-caption recordings. |
| `paired.py` | Read prepared editing pairs for training and audio records for validation/test directly from ZIP. |
| `balanced_sampler.py` | Sample balanced six-task batches for paired Selective Recovery. |
| `worker_determinism.py` | Set reproducible per-epoch, per-rank, and per-worker sampling seeds. |

## Prepare Complex data

**Training:** follow [Source audio and preparation](../../../../pretrain_and_standard_FM_finetune/dataset/README.md) using the paper's source audio or your own audio–caption data; the loader constructs editing examples online. **Evaluation:** use Complex audio editing data, with `complex_val` for validation and `complex_test` for final evaluation.

### Online training sources

Follow the [Source audio and preparation](../../../../pretrain_and_standard_FM_finetune/dataset/README.md) through its data check. It starts with full audio files and category labels / detailed captions, annotates sounding times with energy detection plus CLAP, and writes `data/my_dataset/prepared/online_data.json`.

**Expected input to training:** that config and its referenced source files. The online loader constructs editing instructions and input/target audio during sampling.

### Paired validation and test

[Download `complex_val` and `complex_test` from Hugging Face](../../../README.md#download-complex-validation-and-test-data). Both splits include inputs, instructions, and ground-truth target audio, ready for full Complex metrics. The [metadata](https://huggingface.co/datasets/GeWu-Lab/SteerSound/blob/main/complex_val/metadata.parquet) points to both input and target WAVs inside each ZIP.

You can also use your own paired collection with the layout and schema below.

**Why:** Complex metrics compare each generated edit with its reference target. Keep evaluation recordings separate from training sources.

**Expected layout:**

```text
data/
├── complex_val/
│   ├── metadata.parquet
│   ├── audio.zip
│   └── manifest.json
└── complex_test/           # same three files; optional held-out test set
```

Each table row describes one edit. Store its input and reference-target WAVs inside the ZIP. Both audio files must be mono, 16 kHz, 10 seconds, and finite. ZIP member names must match the paths in the table; keep the ZIP zipped.

| Field | Meaning |
|---|---|
| `sample_id` | Unique record ID within the split. |
| `split` | `val` for validation or `test` for test. |
| `task`, `task_id` | `add=0`, `remove=1`, `generate=2`, `replace=3`, `volume=4`, `pitch=5`. |
| `instruction` | Plain editing instruction matching the input/target pair. |
| `input_audio_path` | Relative ZIP member name for the input WAV; can be empty for Generate, which uses silence. |
| `target_audio_path` | Relative ZIP member name for the reference target WAV; required for every task. |
| `change_vol` | Requested volume change in dB, or `0.0` for other tasks. |
| `change_Pitch` | Requested pitch change in semitones, or `0.0` for other tasks. |
| `base_labels` | Labels of background sounds, as a list. |
| `add_labels` | Labels/descriptions of added or generated sounds, or `[]`. |
| `remove_labels` | Labels of removed/replaced sounds, or `[]`. |
| `start_time`, `end_time` | Edit region in seconds; use `0.0, 10.0` for a full-duration edit. |

For example, one Parquet record shown as JSON:

```json
{
  "sample_id": "remove_000001",
  "split": "val",
  "task": "remove",
  "task_id": 1,
  "instruction": "Remove the sound of a dog barking.",
  "input_audio_path": "audios/val/remove_000001_input.wav",
  "target_audio_path": "audios/val/remove_000001_target.wav",
  "change_vol": 0.0,
  "change_Pitch": 0.0,
  "base_labels": ["rain"],
  "add_labels": [],
  "remove_labels": ["dog barking"],
  "start_time": 0.0,
  "end_time": 10.0
}
```

Here the input contains rain and dog barking, and the reference target contains the rain after removal. This row illustrates the format. The evaluation collection must include **all six tasks** because selection uses a six-task score. For Replace, put the source sound in `remove_labels` and the replacement sound in `add_labels`.

**Expected result:** `complex_val/metadata.parquet` has `split="val"`, unique IDs, and nonempty reference-target paths; every referenced WAV is present in `complex_val/audio.zip`. Test uses the same structure in `complex_test` with `split="test"`.

After preparing it, return to the [training guide](../README.md#3-measure-the-starting-checkpoint) to measure the released checkpoint on this validation set. The resulting `summary.json` is the reference used by `--baseline-val`; no separate base editor weights are required.
