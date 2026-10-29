# Dataset loaders

| File | Purpose |
|---|---|
| `dataset.py` | Construct editing instructions and input/target audio online from labeled source recordings and detailed-caption recordings. |
| `paired.py` | Read prepared editing pairs for training and audio records for validation/test directly from ZIP. |
| `balanced_sampler.py` | Sample balanced six-task batches for paired Selective Recovery. |
| `worker_determinism.py` | Set reproducible per-epoch, per-rank, and per-worker sampling seeds. |

Use the released [SteerSound Selective audio editing data](https://huggingface.co/datasets/GeWu-Lab/SteerSound) with `--train`, `--val`, and `--test`, as shown in the [local training command](../README.md). For online construction from your audio–caption pairs, follow the [Source audio and preparation](../../complex/dataset/README.md); that workflow evaluates on Complex.

## Released Selective data

Download the three Selective folders from the [🤗 dataset](https://huggingface.co/datasets/GeWu-Lab/SteerSound), following the [local download command](../README.md#1-download-weights-and-data).

| Split | Records | Contents | Role |
|---|---:|---|---|
| `selective_train` | 61,440 | Instructions and paired input/target audio; 10,240 records per task. | Supervised training with `--train`. |
| `selective_val` | 6,000 | Input audio and evaluation annotations; 1,000 records per task. | Select the editor using the Selective success rules. |
| `selective_test` | 6,000 | Input audio and evaluation annotations; 1,000 records per task. | Evaluate after selection. |

Each folder contains `metadata.parquet`, `audio.zip`, and `manifest.json`. Paths in the metadata refer to WAV members inside that folder's ZIP. Keep the ZIP zipped. Validation and test work with the published input-only audio under the Selective evaluation rules.

**Expected result:** place these folders under the repository-root `data/` directory, so the training command can use `../../../data/selective_train`, `../../../data/selective_val`, and `../../../data/selective_test` from this entry point.
