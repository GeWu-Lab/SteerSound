"""Device validation for standard PyTorch execution."""
def configure_device(device):
 import torch
 result=torch.device(device)
 if result.type not in ('cpu','cuda'):raise ValueError('Use cpu or an NVIDIA CUDA device')
 return str(result)
