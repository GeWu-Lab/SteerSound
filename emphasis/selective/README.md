# Emphasis — selective

This is a standalone CUDA training program. Install its local requirements-training.txt and requirements-metrics.txt after installing CUDA PyTorch as documented in the repository README. Run commands from this folder.

```bash
torchrun --standalone --nproc_per_node=16 train.py --model GeWu-Lab/SteerSound --dit base_editor.safetensors --train /path/to/data/selective_train --val /path/to/data/selective_val --test /path/to/data/selective_test --output outputs/emphasis_selective
```

`--dit` must identify the separately pretrained base editor; it can be an inference safetensors file or a trusted native DiT training state. It does not restore optimizer state. The final published Recovery model is used for compatible frozen VAE/T5/vocoder assets; its DiT is replaced by your supplied initialization.

Download `selective_train`, `selective_val` and `selective_test` from the data Hub; leave audio.zip intact. The training set has paired targets; validation/test are scored using the original success rules.

Global batch defaults to 30 × GPU count × accumulation; 16 GPUs yields 480. For 4 GPUs, add `--accumulation 4` to retain 480. Changing microbatch or hardware can change runtime/memory; CUDA was not available for release testing.

Selected weights: `output/selected.safetensors`; validation scores and identity: `history.json` / `selection.json`; DeepSpeed epoch states: `checkpoint_*`; optional final test: `test_final_once`. Output directories must be new; this entry point starts fresh and does not resume partial training.

Control prefixes are added by the model text encoder, not stored in dataset instructions. Metric implementations are copied locally. See the root README for scientific settings, CLAP download and metric definitions.
