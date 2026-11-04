# Pretrain — complex

This is a standalone CUDA training program. Install its local requirements-training.txt and requirements-metrics.txt after installing CUDA PyTorch as documented in the repository README. Run commands from this folder.

```bash
torchrun --standalone --nproc_per_node=16 train.py --model GeWu-Lab/SteerSound --online-config online_data.json --output outputs/pretrain
```

Copy `online_data.example.json` to `online_data.json` and supply the original timing parquet, raw audio, long-prompt CSV/audio and label dictionary. Those external raw training assets are not part of the published editing dataset. Pretraining initializes a fresh DiT; `--model` supplies only architecture and frozen components. It saves every 5 epochs and the final epoch, with no validation/test selection.

Control prefixes are added by the model text encoder, not stored in dataset instructions. Metric implementations are copied locally. See the root README for scientific settings, CLAP download and metric definitions.
