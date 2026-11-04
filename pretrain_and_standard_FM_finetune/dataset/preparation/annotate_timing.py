#!/usr/bin/env python3
"""Annotate sounding regions using relative energy candidates followed by CLAP.

Adapted from the original SteerSound detect_worker.py and postprocess_clamp.py.
Uses sound activity (not speech-only VAD); no source separation is performed.
"""
import argparse
from pathlib import Path

from common import SR, normalize, read_table, write_table


def candidates(wave, db_threshold=-15):
    import librosa
    import numpy as np

    if not wave.size or np.max(np.abs(wave)) == 0:
        return []
    rms = librosa.feature.rms(y=wave, frame_length=800, hop_length=400)[0]
    active = librosa.power_to_db(rms ** 2, ref=np.max, top_db=None) > db_threshold
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s * 400), int(min(e * 400, len(wave)) - s * 400))
            for s, e in zip(starts, ends) if min(e * 400, len(wave)) - s * 400 >= 8000]


class ClapFilter:
    def __init__(self, checkpoint, device):
        import torch
        from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel

        self.torch = torch
        self.device = device
        self.model = ClapModel.from_pretrained(checkpoint).to(device).eval()
        self.feature = AutoFeatureExtractor.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.text_cache = {}

    def scores(self, segments, text):
        import torch.nn.functional as F
        import torchaudio.functional as AF

        torch = self.torch
        with torch.no_grad():
            if text not in self.text_cache:
                tokens = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
                embedding = self.model.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()})
                # Current and older transformers releases both supported.
                if not isinstance(embedding, torch.Tensor):
                    embedding = embedding.pooler_output
                # A bounded cache avoids retaining every unique long caption.
                if len(self.text_cache) >= 4096:
                    self.text_cache.clear()
                self.text_cache[text] = F.normalize(embedding, dim=-1)
            audio = [AF.resample(torch.from_numpy(s), SR, 48000).numpy() for s in segments]
            inputs = self.feature(audio, sampling_rate=48000, return_tensors="pt")
            embedding = self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
            if not isinstance(embedding, torch.Tensor):
                embedding = embedding.pooler_output
            return (F.normalize(embedding, dim=-1) @ self.text_cache[text].T).flatten().cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="CSV/Parquet with path and label or caption")
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--kind", choices=("labels", "captions"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clap-model", default="laion/clap-htsat-fused")
    p.add_argument("--device", default="cpu")
    p.add_argument("--db-threshold", type=float, default=-15)
    p.add_argument("--clap-threshold", type=float, default=.1)
    p.add_argument("--batch-size", type=int, default=32, help="Candidate regions scored per CLAP batch")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1, help="Contiguous disjoint shards; use separate output files")
    args = p.parse_args()
    if args.batch_size < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("Invalid batch size or shard index")
    if args.output.exists():
        p.error("Output already exists; choose a new file")
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    source = read_table(args.input)
    source = source.iloc[np.array_split(np.arange(len(source)), args.num_shards)[args.shard_index]]
    table = normalize(source, args.audio_root, args.kind)
    clap = ClapFilter(args.clap_model, args.device)
    text_column = "label" if args.kind == "labels" else "caption"
    for count, (index, row) in enumerate(table.iterrows(), 1):
        filename = args.audio_root / row.path
        wave, rate = sf.read(filename, dtype="float32", always_2d=True)
        wave = wave[:, 0].copy()  # Original pipeline uses the first channel.
        if not np.isfinite(wave).all():
            raise ValueError("Non-finite audio: " + str(filename))
        if rate != SR:
            wave = AF.resample(torch.from_numpy(wave), rate, SR).numpy()
        regions = candidates(wave, args.db_threshold)
        accepted = []
        for offset in range(0, len(regions), args.batch_size):
            batch = regions[offset:offset + args.batch_size]
            segments = []
            for s, length in batch:
                # CLAP sees the center 10 s of long candidates; stored region stays full length.
                center_start = s + max(0, (length - 10 * SR) // 2)
                segments.append(wave[center_start:center_start + min(length, 10 * SR)])
            scores = clap.scores(segments, row[text_column])
            accepted.extend(region for region, score in zip(batch, scores) if score >= args.clap_threshold)
        points = [pair for pair in accepted if pair[1] < SR]
        pieces = [pair for pair in accepted if pair[1] >= SR]
        table.at[index, "point_audio"] = str(points) if points else None
        table.at[index, "piece_audio"] = str(pieces) if pieces else None
        if count == 1 or count % 100 == 0 or count == len(table):
            print(f"Annotated {count}/{len(table)}", flush=True)
    write_table(table, args.output)
    print(f"Saved {args.output}. Empty lists mean no accepted event, not a full-file annotation.")


if __name__ == "__main__":
    main()
