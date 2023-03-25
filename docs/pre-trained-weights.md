{%
laika.title="Converting pre-trained weights"
%}

# Converting pre-trained weights from PyTorch

Loading pre-trained weights from PyTorch into Storch is an important feature for transfer-learning,
or for doing inference on models trained with PyTorch.

Currently, this is not as simple as it should be because serialized weights cannot be loaded into Storch if they
contain weights stored as `torch.nn.parameter.Parameter`, a subclass of `torch.Tensor`.
We have to convert these parameters to regular tensors first to be able to load them in Storch.

To help with this task, we provide a simple [conversion script](https://github.com/sbrunk/storch/blob/main/scripts/convert-weights/convert_weights.py).
Currently the script only converts pre-trained ResNet weights but it shouldn't be too difficult to apply it to other weights as well.

The [converted weights](https://github.com/sbrunk/storch/releases/tag/pretrained-weights) are also available for download
from the Storch GitHub repository.

We hope to improve the situation by creating our own reader that allows direct loading of PyTorch weights.
