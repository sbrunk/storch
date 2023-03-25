# Script for converting weights of class `torch.nn.parameter.Parameter` to regular tensors which enables loading them from libtorch

from pathlib import Path
from urllib.parse import urlparse
from torchvision.models.resnet import *
import torch

model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

for weights_enum in [ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights]:
    for weights in weights_enum:
        parts = urlparse(weights.url)
        filename = Path(parts.path).name
        print(f"Converting {filename}")
        state_dict = weights.get_state_dict(progress=True)
        # Convert from torch.nn.Parameter to torch.Tensor so we can load the state dict from libtorch
        converted_state_dict = {k: v.clone().detach()
                                for k, v in state_dict.items()}
        torch.save(converted_state_dict, (model_path / filename))
