# Model size and GPU memory

| Component | Parameters |
|---|---:|
| DiT editor | 0.673B |
| Audio VAE | 0.509B |
| T5 encoder | 0.110B |
| HiFi-GAN vocoder | 0.055B |
| Total | **1.348B** |

**GPU memory for inference: approximately 8 GB** (BF16, batch size 1, 10-second audio).
