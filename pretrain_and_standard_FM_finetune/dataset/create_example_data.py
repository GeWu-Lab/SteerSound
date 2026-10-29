"""Create a small synthetic dataset that demonstrates the pretraining input format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing files are never overwritten.")

    output.mkdir(parents=True)
    for directory in ("raw_audio", "long_prompt_audio"):
        (output / directory).mkdir()
    sample_rate = 16000
    time = np.arange(10 * sample_rate, dtype=np.float32) / sample_rate
    main_rows, long_rows, labels = [], [], {}
    for index in range(8):
        frequency = 220 + index * 110
        name = f"tone_{index:02d}.wav"
        label = f"Synthetic {frequency} Hz tone"
        caption = f"a steady synthetic {frequency} Hz tone"
        wave = (0.1 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
        active = ((time >= 1) & (time < 1.75)) | ((time >= 3) & (time < 8))
        wave[~active] = 0
        for directory in ("raw_audio", "long_prompt_audio"):
            sf.write(output / directory / name, wave, sample_rate, subtype="PCM_16")
        timing = {"point_audio": "[(16000, 12000)]", "piece_audio": "[(48000, 80000)]"}
        main_rows.append({
            "path": name, "duration": 10.0, "label": label, **timing,
        })
        long_rows.append({"path": name, "duration": 10.0, "label": caption, **timing})
        labels[label] = [caption, f"a synthetic {frequency} Hz tone"]

    parquet = output / "train_with_timing.parquet"
    csv = output / "long_prompt_merged.csv"
    dictionary = output / "label_dict.pt"
    pd.DataFrame(main_rows).to_parquet(parquet, index=False)
    pd.DataFrame(long_rows).to_csv(csv, index=False)
    torch.save(labels, dictionary)

    template = Path(__file__).resolve().parents[1] / "online_data.example.json"
    config = json.loads(template.read_text())
    config.update(
        parquet_path=str(parquet), audio_base_path=str(output / "raw_audio"),
        long_prompt_csv_path=str(csv), long_prompt_audio_base_path=str(output / "long_prompt_audio"),
        label_dict_path=str(dictionary),
    )
    config_path = output / "online_data.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Created synthetic format examples in {output}")
    print(f"Configuration: {config_path}")


if __name__ == "__main__":
    main()
