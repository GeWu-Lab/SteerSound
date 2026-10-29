# GPU inference

Run from this folder. Linux with Python 3.10/3.11 and an NVIDIA GPU is the installation target. CUDA execution and VRAM were not measured on the release host.

```bash
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

No package installation is needed when running infer.py here. From the GitHub repository root, `pip install -e . --no-deps` also exposes the `steersound` Python API. Default inference is BF16, 50-step Euler, 10 seconds at 16 kHz. Supply a plain instruction; task ID and changes are inferred internally.

```bash
python infer.py --instruction "Remove the sound of a dog barking." --input input.wav --output edited.wav
```

| Task | Plain instruction | Numeric conditioning |
|---|---|---|
| Generation | `Generate the sound of rain.` | None |
| Addition | `Add the sound of a dog barking.` | None |
| Removal | `Remove the sound of a dog barking.` | None |
| Replacement | `Replace the sound of a dog barking by the sound of a cat meowing.` | None |
| Volume | `Increase the volume of the sound of a dog barking by 3.0 dB.` | +3 dB |
| Pitch | `Reduce the pitch of the sound of a dog barking by 2.0 semitones.` | −2 semitones |

Use `Increase` or `Reduce` and an explicit nonnegative magnitude for pitch/volume. Volume supports −5 to +5 dB; pitch supports −4 to +4 semitones in steps of 0.5. Native dataset instructions omit units; units in user input are converted internally to the native one-decimal, unit-free text template. For temporal editing, place the region before the amount, e.g. `Increase the volume of the sound of a dog barking from 2.0 to 4.0 seconds by 3.0 dB.` Use one operation per instruction. This deterministic parser supports the documented English templates; it does not use another language model or guess missing amounts. Unsupported or ambiguous instructions raise an error. Time intervals are conveyed through text to the editor, rather than hard masks applied to the output.

See the root README and MEMORY.md for installation, full evaluation and GPU memory requirements.


Python API, from this directory:

```python
import soundfile as sf
from steersound import SteerSoundPipeline
pipe = SteerSoundPipeline.from_pretrained("GeWu-Lab/SteerSound", device="cuda")
wave = pipe("Remove the sound of a dog barking.", audio="input.wav")
sf.write("edited.wav", wave, pipe.sample_rate, subtype="PCM_16")
```

For complete split evaluation, install requirements-metrics.txt, download laion/larger_clap_general to ~/.cache/steersound/clap, then run:

```bash
python evaluate.py --data /path/to/private_complex_val --setting complex --split val --output outputs/val
```

An absolute summary without --baseline can serve as the base editor's normalization reference. Use `--dit base_editor.safetensors` to evaluate a specific base editor instead of the final Recovery DiT. Subsequent comparisons pass `--baseline /path/to/base_val/summary.json`. Keep validation and test baselines separate. See MEMORY.md for model parameter counts and total GPU memory requirements.

All public val/test splits omit target audio. Full Complex reference-based evaluation requires a privately held paired dataset in the same metadata/ZIP layout; public splits can still be used for instruction-based inference. Selective task-success evaluation uses the public input-only data.
