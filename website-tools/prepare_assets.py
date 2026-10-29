"""Prepare the supplied paper figures and unmodified WAV demos for the project page."""
import argparse
import array
import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from PIL import Image
from build_data_figures import build as build_data_figures

parser = argparse.ArgumentParser()
parser.add_argument('--paper-dir', type=Path)
parser.add_argument('--demo-dir', type=Path, required=True)
parser.add_argument('--audio-only', action='store_true',
                    help='Refresh demos without rebuilding paper figures')
args = parser.parse_args()
if not args.audio_only and args.paper_dir is None:
    parser.error('--paper-dir is required unless --audio-only is set')
site = Path(__file__).resolve().parents[1] / 'website'
for folder in ['assets/audio', 'assets/figures', 'assets/fonts', 'data']:
    (site / folder).mkdir(parents=True, exist_ok=True)

figures = {
    'overview': 'figure200', 'interventions': 'figure102',
    'early-selection': 'figure103', 'emphasis': 'figure53',
    'recovery': 'figure71', 'architecture': 'figure72',
    'data': 'figure54', 'robustness': 'figure503',
    'human-evaluation': 'figure501',
}
if not args.audio_only:
    for name, source in figures.items():
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / name
            subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2800', '-png',
                            str(args.paper_dir / 'figures' / f'{source}.pdf'), str(output)],
                           check=True, capture_output=True)
            with Image.open(output.with_suffix('.png')) as image:
                image.convert('RGB').save(site / 'assets/figures' / f'{name}.webp',
                                          'WEBP', quality=94, method=6)
    build_data_figures(args.paper_dir)

samples = [
    ('remove1', 'remove', 'Background music', 'source.wav', 'edited.wav'),
    ('remove2', 'remove', 'Speaking voice', 'source.wav', 'edited.wav'),
    ('replace1', 'replace', 'Ringtone → drum', 'source.wav', 'edited.wav'),
    ('replace2', 'replace', 'Piano → violin', 'source.wav', 'edited.wav'),
    ('add1', 'add', 'Piano', 'reference_segment.wav', 'steersound_result 2.wav'),
    ('add2', 'add', 'Rain', 'source.wav', 'edited.wav'),
    # Present both adjustment tasks in Increase → Decrease order.
    ('pitch1', 'pitch', 'Violin · +4 semitones', 'source.wav', 'edited.wav'),
    # The author confirmed target.wav is the model output on 2026-10-26.
    ('pitch2', 'pitch', 'Shouting voice · −4 semitones', 'source.wav', 'target.wav'),
    ('volume2', 'volume', 'Background music · +5 dB', 'source.wav', 'edited.wav'),
    ('volume1', 'volume', 'Sharpening · −5 dB', 'source.wav', 'edited.wav'),
    ('generate1', 'generate', 'Dog barking & baby crying', None, 'edited.wav'),
    ('generate2', 'generate', 'Violin', None, 'edited.wav'),
]


def inspect_audio(path):
    with wave.open(str(path), 'rb') as audio:
        assert audio.getsampwidth() == 2, 'Expected 16-bit PCM input'
        values = array.array('h', audio.readframes(audio.getnframes()))
        channels = audio.getnchannels()
        mono = [sum(values[i:i + channels]) / channels / 32768
                for i in range(0, len(values), channels)]
        step = max(1, len(mono) // 160)
        # Windowed RMS makes a stable waveform preview; audio bytes stay unchanged.
        peaks = [math.sqrt(sum(x * x for x in mono[i:i + step]) /
                           max(1, len(mono[i:i + step])))
                 for i in range(0, len(mono), step)][:160]
        return audio.getnframes() / audio.getframerate(), peaks


manifest = []
for folder, task, title, original, edited in samples:
    item = {'id': folder, 'task': task, 'title': title,
            'instruction': (args.demo_dir / folder / 'instruction.txt').read_text().strip()}
    waves = []
    for role, file in [('original', original), ('edited', edited)]:
        if file is None:
            continue
        source = args.demo_dir / folder / file
        relative = f'assets/audio/{folder}-{role}.wav'
        shutil.copyfile(source, site / relative)
        duration, peaks = inspect_audio(source)
        item[role] = {'src': relative, 'duration': round(duration, 3), 'peaks': peaks}
        waves.extend(peaks)
    scale = max(waves) or 1
    for role in ['original', 'edited']:
        if role in item:
            item[role]['peaks'] = [round(x / scale, 4) for x in item[role]['peaks']]
    manifest.append(item)
(site / 'data/demos.json').write_text(json.dumps(manifest, ensure_ascii=False, separators=(',', ':')))
if not args.audio_only:
    print(f'Prepared {len(figures)} paper figures and 3 data SVGs.')
print(f'Prepared {len(manifest)} demos ({sum(1 + ("original" in d) for d in manifest)} WAV files).')
