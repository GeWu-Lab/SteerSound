# Source-data preparation tools

Follow the [complete walkthrough](../README.md) for commands, before/after records, expected files, and fine-tuning. Run commands from the repository root.

**This prepares source audio for online editing-example construction.** Both collections use one full audio file per record. No prebuilt editing input/target pairs are needed; the loader generates them during training.

## Processing order

| Step | Required input before processing | Output after processing | Why |
|---|---|---|---|
| 1. `annotate_timing.py` | Audio files and a CSV/Parquet manifest with `path,label` (`--kind labels`) or `path,caption` (`--kind captions`). Paths are relative to `--audio-root`. No timing columns are needed yet. | One output row per input row: `path,label,duration,point_audio,piece_audio` or `path,caption,duration,point_audio,piece_audio`. Both use full files. | Find where the annotated sound occurs so editing tasks can sample those regions. |
| 2. `prepare_data.py` | The annotated tables with both timing columns, their audio roots, and optional category-expression JSON. | `label_audio.parquet`, optional `long_prompt.csv`, optional `label_dict.pt`, and `online_data.json` with absolute paths. | Convert the input schema into the loader's files and collect paths/settings in one config. |
| 3. `check_data.py` | The generated `online_data.json` and all referenced files. | Pool counts and a final `PASS` line, or an error describing the failed check. No new dataset files. | Catch I/O, timing, and sampling problems before training. |

If your own timing annotations already match the [required format](../README.md#3-annotate-when-the-sound-occurs), start at step 2. The supplied [ready-to-use examples](../examples/README.md#run-the-ready-to-use-examples) already contain such annotations.

## What annotation does

```text
Audio + category/caption
    → energy-based candidate regions (at least 0.5 seconds)
    → CLAP text/audio similarity filtering (default ≥0.1)
    → accepted regions
        ├── point_audio: 0.5 to <1 second
        └── piece_audio: at least 1 second
```

The command performs candidate detection and CLAP filtering together. CLAP compares each candidate region with its record's category or caption. It filters **regions**, not whole manifest rows. Rejected regions are absent from the output timings; if none pass, the fields are blank and the row remains. Source audio files are unchanged.

Timing values are lists of **(start sample, length in samples)** at 16 kHz. For both collections, sample 0 is the start of the full file, and `duration` is its full length in seconds. See the walkthrough for [CSV records before and after annotation](../README.md#3-annotate-when-the-sound-occurs).

## How to recognize completion

| Command | Successful terminal output |
|---|---|
| `annotate_timing.py` | `Annotated N/N` followed by `Saved <output path>`. |
| `prepare_data.py` | Pool counts followed by `Training config: <absolute path>/online_data.json`. |
| `check_data.py` | Pool counts followed by `PASS: files, schema, duration, 16 kHz timing bounds, and sampling pools`. |

Empty timing fields are valid for individual records. A pool needed by the selected training setup must still be nonempty; see the [check and troubleshooting table](../README.md#5-check-the-data). Preparation runs those checks before creating its output directory. All output paths must be new.

## Options for larger collections

Annotation supports `--num-shards N --shard-index I` (`I` is zero-based). Run shards with separate output filenames/devices, then concatenate the output tables in index order before preparation. Each record belongs to exactly one contiguous shard. The model loads once per process; `--batch-size` controls candidate regions per CLAP batch (default 32).

The annotation code adapts the original SteerSound `detect_worker.py` and `postprocess_clamp.py`: energy candidates and CLAP filtering are retained, while paths/devices are configurable, silent recordings yield no events, whole files are read, and intervals are clamped to the decoded waveform.

Run `python pretrain_and_standard_FM_finetune/dataset/preparation/<tool>.py --help` for options.
