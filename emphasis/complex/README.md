# Emphasis — complex

This is a standalone CUDA training program. Install its local requirements-training.txt and requirements-metrics.txt after installing CUDA PyTorch as documented in the repository README. Run commands from this folder.

```bash
torchrun --standalone --nproc_per_node=16 train.py --model GeWu-Lab/SteerSound --dit base_editor.safetensors --online-config online_data.json --val /path/to/private_complex_val --test /path/to/private_complex_test --baseline-val /path/to/base_val/summary.json --baseline-test /path/to/base_test/summary.json --output outputs/emphasis_complex
```

`--dit` must identify the separately pretrained base editor; it can be an inference safetensors file or a trusted native DiT training state. It does not restore optimizer state. The final published Recovery model is used for compatible frozen VAE/T5/vocoder assets; its DiT is replaced by your supplied initialization.

Copy `online_data.example.json` and set actual raw corpus paths. The full online Complex training corpus is external to this data release.

Generate separate absolute base-model summaries for validation and test using `inference/evaluate.py --dit base_editor.safetensors`. Pass their summary.json files as shown; the validation TBG denominator must be the base editor, not the Emphasis or final Recovery model. Test is never used for checkpoint selection.

Global batch defaults to 30 × GPU count × accumulation; 16 GPUs yields 480. For 4 GPUs, add `--accumulation 4` to retain 480. Changing microbatch or hardware can change runtime/memory; CUDA was not available for release testing.

Selected weights: `output/selected.safetensors`; validation scores and identity: `history.json` / `selection.json`; DeepSpeed epoch states: `checkpoint_*`; optional final test: `test_final_once`. Output directories must be new; this entry point starts fresh and does not resume partial training.

Control prefixes are added by the model text encoder, not stored in dataset instructions. Metric implementations are copied locally. See the root README for scientific settings, CLAP download and metric definitions.

Published Complex val/test contain input audio only. Full reference-based validation/test metrics and TBG require your private paired ground-truth datasets; use those private folders in the command above. The original metric definitions and validation checkpoint-selection protocol are unchanged.
