#!/usr/bin/env python3
"""Preflight prepared source data without loading training/model weights."""
import argparse
import json
from pathlib import Path

from common import normalize, read_table, validate_pools


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--decode", action="store_true", help="Also decode every referenced full audio file and check finite samples")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    labels = normalize(read_table(config["parquet_path"]), config["audio_base_path"], "labels",
                       require_timing=True, decode=args.decode)
    captions = None
    if config.get("long_prompt_csv_path") and config.get("long_prompt_prob", 0) > 0:
        captions = read_table(config["long_prompt_csv_path"]).rename(columns={"label": "caption"})
        captions = normalize(captions, config["long_prompt_audio_base_path"], "captions",
                             require_timing=True, decode=args.decode)
    aliases = None
    if config.get("label_dict_path"):
        import torch
        aliases = torch.load(config["label_dict_path"], map_location="cpu", weights_only=True)
    print(json.dumps(validate_pools(config, labels, captions, aliases), indent=2))
    print("PASS: files, schema, duration, 16 kHz timing bounds, and sampling pools")


if __name__ == "__main__":
    main()
