#!/usr/bin/env python3
"""Convert annotated category/caption manifests into an online training config."""
import argparse
import json
from pathlib import Path

from common import normalize, read_table, validate_pools


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True, help="Timed category-level CSV/Parquet")
    p.add_argument("--label-audio-root", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--captions", type=Path, help="Timed detailed-caption CSV/Parquet")
    group.add_argument("--labels-only", action="store_true", help="Explicitly draw all targets from category-level audio")
    p.add_argument("--caption-audio-root", type=Path)
    p.add_argument("--aliases", type=Path, help="Optional JSON object: category -> alternative category phrases")
    p.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[2] / "online_data.example.json")
    p.add_argument("--task-weights", nargs=6, type=float, metavar=("ADD", "REMOVE", "GENERATE", "REPLACE", "VOLUME", "PITCH"))
    p.add_argument("--output", type=Path, required=True, help="New preparation directory")
    args = p.parse_args()
    if args.captions and args.caption_audio_root is None:
        p.error("--captions requires --caption-audio-root")
    if args.output.exists():
        p.error("Output already exists; choose a new directory")
    labels = normalize(read_table(args.labels), args.label_audio_root, "labels", require_timing=True)
    captions = None if args.labels_only else normalize(read_table(args.captions), args.caption_audio_root,
                                                       "captions", require_timing=True)
    aliases = None if args.aliases is None else json.loads(args.aliases.read_text())
    if aliases is not None and not isinstance(aliases, dict):
        p.error("--aliases must contain a JSON object")
    config = json.loads(args.template.read_text())
    if args.task_weights is not None:
        config["task_probs"] = args.task_weights
    if args.labels_only:
        config.update(long_prompt_prob=0.0, long_prompt_tasks=[], required_long_prompt_tasks=[],
                      long_prompt_csv_path=None, long_prompt_audio_base_path=None)
    else:
        config.update(long_prompt_prob=1.0, long_prompt_tasks=[0, 2], required_long_prompt_tasks=[0, 2, 3])
    counts = validate_pools(config, labels, captions, aliases)
    output = args.output.resolve()
    # Complete validation before creating an output folder.
    output.mkdir(parents=True)
    label_columns = ["path", "duration", "label", "point_audio", "piece_audio"]
    labels[label_columns].to_parquet(output / "label_audio.parquet", index=False)
    config.update(parquet_path=str(output / "label_audio.parquet"),
                  audio_base_path=str(args.label_audio_root.resolve()), label_dict_path=None)
    if captions is not None:
        # Legacy loader calls the detailed-caption column 'label'; the input schema calls it 'caption'.
        caption_columns = ["path", "duration", "caption", "point_audio", "piece_audio"]
        captions[caption_columns].rename(columns={"caption": "label"}).to_csv(output / "long_prompt.csv", index=False)
        config.update(long_prompt_csv_path=str(output / "long_prompt.csv"),
                      long_prompt_audio_base_path=str(args.caption_audio_root.resolve()))
    if aliases is not None:
        import torch
        torch.save(aliases, output / "label_dict.pt")
        config["label_dict_path"] = str(output / "label_dict.pt")
    (output / "online_data.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print("Training config:", output / "online_data.json")


if __name__ == "__main__":
    main()
