# Modules

Storch provides a neural network module API with building blocks for creating stateful neural network architectures.

## Simple custom module example

```scala mdoc:invisible
torch.manualSeed(0)
```

```scala mdoc
import torch.*
import torch.nn
import torch.nn.functional as F

class LeNet[D <: BFloat16 | Float32: Default] extends nn.Module:
  val conv1 = register(nn.Conv2d(1, 6, 5))
  val conv2 = register(nn.Conv2d(6, 16, 5))
  val fc1 = register(nn.Linear(16 * 4 * 4, 120))
  val fc2 = register(nn.Linear(120, 84))
  val fc3 = register(nn.Linear(84, 10))

  def apply(i: Tensor[D]): Tensor[D] =
    var x = F.maxPool2d(F.relu(conv1(i)), (2, 2))
    x = F.maxPool2d(F.relu(conv2(x)), 2)
    x = x.view(-1, 16 * 4 * 4)
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = fc3(x)
    x
```

```scala mdoc
val model = LeNet()
val input = torch.rand(Seq(1, 1, 28, 28))
model(input)
```