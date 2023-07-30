Learn the Basics ||
Quickstart ||
[Tensors](tensors.md) ||
Datasets & DataLoaders ||
Transforms ||
**Build Model** ||
[Autograd](autograd.md) ||
Optimization ||
Save & Load Model

```scala mdoc:invisible
torch.manualSeed(0)
```

# Build the Neural Network

Neural networks comprise of layers/modules that perform operations on data.
The @:api(torch.nn) namespace provides all the building blocks you need to
build your own neural network. Every module in PyTorch subclasses the @:api(torch.nn.modules.Module).
A neural network is a module itself that consists of other modules (layers). This nested structure allows for
building and managing complex architectures easily.

In the following sections, we'll build a neural network to classify images in the FashionMNIST dataset.

```scala mdoc
import torch.*
```

## Get Device for Training
We want to be able to train our model on a hardware accelerator like the GPU,
if it is available. Let's check to see if
[torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) is available, else we
continue to use the CPU.


```scala mdoc
import torch.Device.{CPU, CUDA}
val device = if torch.cuda.isAvailable then CUDA else CPU
println(s"Using $device device")
```

## Define the Class
We define our neural network by subclassing ``nn.Module``, and
initialize the neural network layers in the constructor. Every ``nn.Module`` subclass implements
the operations on input data in the ``apply`` method.

```scala mdoc
class NeuralNetwork extends nn.Module:
  val flatten = nn.Flatten()
  val linearReluStack = register(nn.Sequential(
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
  ))
  
  def apply(x: Tensor[Float32]) =
    val flattened = flatten(x)
    val logits = linearReluStack(flattened)
    logits
```

We create an instance of ``NeuralNetwork``, and move it to the ``device``, and print
its structure.

```scala mdoc
val model = NeuralNetwork().to(device)
println(model)
```

To use the model, we pass it the input data. This executes the model's ``apply`` method.

Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output.
We get the prediction probabilities by passing it through an instance of the ``nn.Softmax`` module.


```scala mdoc
val X = torch.rand(Seq(1, 28, 28), device=device)
val logits = model(X)
val predProbab = nn.Softmax(dim=1).apply(logits)
val yPred = predProbab.argmax(1)
println(s"Predicted class: $yPred")
```

--------------


## Model Layers

Let's break down the layers in the FashionMNIST model. To illustrate it, we
will take a sample minibatch of 3 images of size 28x28 and see what happens to it as
we pass it through the network.


```scala mdoc
val inputImage = torch.rand(Seq(3,28,28))
print(inputImage.size)
```

### nn.Flatten
We initialize the @:api(torch.nn.modules.flatten.Flatten)
layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (
the minibatch dimension (at dim=0) is maintained).


```scala mdoc
val flatten = nn.Flatten()
val flatImage = flatten(inputImage)
println(flatImage.size)
```

### nn.Linear
The @:api(torch.nn.modules.linear.Linear) layer
is a module that applies a linear transformation on the input using its stored weights and biases.


```scala mdoc
val layer1 = nn.Linear(inFeatures=28*28, outFeatures=20)
var hidden1 = layer1(flatImage)
println(hidden1.size)
```

### nn.ReLU
Non-linear activations are what create the complex mappings between the model's inputs and outputs.
They are applied after linear transformations to introduce *nonlinearity*, helping neural networks
learn a wide variety of phenomena.

In this model, we use @:api(torch.nn.modules.activation.ReLU) between our
linear layers, but there's other activations to introduce non-linearity in your model.


```scala mdoc:nest
println(s"Before ReLU: $hidden1\n\n")
val relu = nn.ReLU()
hidden1 = relu(hidden1)
println(s"After ReLU: $hidden1")
```

### nn.Sequential
@:api(torch.nn.modules.container.Sequential) is an ordered
container of modules. The data is passed through all the modules in the same order as defined. You can use
sequential containers to put together a quick network like ``seqModules``.


```scala mdoc
val seqModules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
val inputImage = torch.rand(Seq(3,28,28))
val logits = seqModules(inputImage)
```

### nn.Softmax
The last linear layer of the neural network returns `logits` - raw values in $\[-\infty, \infty\]$ - which are passed to the
@:api(torch.nn.modules.activation.Softmax) module. The logits are scaled to values
\[0, 1\] representing the model's predicted probabilities for each class. ``dim`` parameter indicates the dimension along
which the values must sum to 1.


```scala mdoc
val softmax = nn.Softmax(dim=1)
val predProbab = softmax(logits)
```

## Model Parameters
Many layers inside a neural network are *parameterized*, i.e. have associated weights
and biases that are optimized during training. Subclassing ``nn.Module`` automatically
tracks all fields defined inside your model object, and makes all parameters
accessible using your model's ``parameters()`` or ``namedParameters()`` methods.

In this example, we iterate over each parameter, and print its size and a preview of its values.


```scala mdoc
println(s"Model structure: ${model}")

for (name, param) <- model.namedParameters() do
    println(s"Layer: ${name} | Size: ${param.size.mkString} | Values:\n${param(Slice(0, 2))} ")
```

--------------


## Further Reading
- torch.@:api(torch.nn) API


