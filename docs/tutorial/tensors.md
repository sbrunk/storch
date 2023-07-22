Learn the Basics ||
Quickstart ||
**Tensors** ||
Datasets & DataLoaders ||
Transforms ||
[Build Model](buildmodel.md) ||
[Autograd](autograd.md) ||
Optimization ||
Save & Load Model

# Tensors

```scala mdoc:invisible
torch.manualSeed(0)
```

Tensors are a specialized data structure that are very similar to arrays
and matrices. In PyTorch, we use tensors to encode the inputs and
outputs of a model, as well as the model’s parameters.

Tensors are similar to [NumPy’s](https://numpy.org/) ndarrays, except
that tensors can run on GPUs or other hardware accelerators. Tensors
are also optimized for automatic differentiation (we'll see more about
that later in the [Autograd](autograd.md) section). If
you’re familiar with ndarrays, you’ll be right at home with the Tensor
API. If not, follow along!

## Initializing a Tensor

Tensors can be initialized in various ways. Take a look at the following
examples:

**Directly from data**

Tensors can be created directly from data. The data type is
automatically inferred.

```scala mdoc
val data = Seq(1, 2, 3, 4)
val xData = torch.Tensor(data).reshape(2,2)
```

**From another tensor:**

The new tensor retains the properties (shape, datatype) of the argument
tensor, unless explicitly overridden.


```scala mdoc
// Ones Tensor:
val xOnes = torch.onesLike(xData) // retains the properties of xData
```

```scala mdoc
// Random Tensor:
val xRand = torch.randLike(xData, dtype=torch.float32) // overrides the datatype of xData
```

**With random or constant values:**

`shape` is a tuple of tensor dimensions. In the functions below, it
determines the dimensionality of the output tensor.

```scala mdoc
val shape = Seq(2,3)

// Random Tensor:
val randTensor = torch.rand(shape)

// Ones Tensor: 
val onesTensor = torch.ones(shape)

// Zeros Tensor:
val zerosTensor = torch.zeros(shape)
```

## Attributes of a Tensor

Tensor attributes describe their shape, datatype, and the device on
which they are stored.

```scala mdoc
var tensor = torch.rand(Seq(3,4))

println(s"Shape of tensor: ${tensor.shape}")
println(s"Datatype of tensor: ${tensor.dtype}")
println(s"Device tensor is stored on: {tensor.device}")
```

------------------------------------------------------------------------

## Operations on Tensors

Over 100 tensor operations, including arithmetic, linear algebra, matrix
manipulation (transposing, indexing, slicing), sampling and more are
comprehensively described
[here](https://pytorch.org/docs/stable/torch.html).

Each of these operations can be run on the GPU (at typically higher
speeds than on a CPU). If you’re using Colab, allocate a GPU by going to
Runtime \> Change runtime type \> GPU.

By default, tensors are created on the CPU. We need to explicitly move
tensors to the GPU using `.to` method (after checking for GPU
availability). Keep in mind that copying large tensors across devices
can be expensive in terms of time and memory! We move our tensor to the
GPU if available

```scala mdoc
if torch.cuda.isAvailable then  
  tensor = tensor.to(torch.Device.CUDA)
```

Try out some of the operations from the list. If you're familiar with
the NumPy API, you'll find the Tensor API a breeze to use.

**Standard numpy-like indexing and slicing:**

```scala mdoc
import torch.{---, Slice}
tensor = torch.ones(Seq(4, 4))
println(s"First row: ${tensor(0)}")
println(s"First column: ${tensor(Slice(), 0)}")
println(s"Last column: ${tensor(---, -1)}")
//tensor(---,1) = 0 TODO update op
println(tensor)
```

**Joining tensors** You can use `torch.cat` to concatenate a sequence of
tensors along a given dimension. See also
[torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
another tensor joining op that is subtly different from `torch.cat`.

```scala mdoc
val t1 = torch.cat(Seq(tensor, tensor, tensor), dim=1)
println(t1)
```

**Arithmetic operations**

```scala mdoc
// This computes the matrix multiplication between two tensors. y1, y2, y3 will
// have the same value
// `tensor.mT` returns the transpose of a tensor
val y1 = tensor `@` tensor.mT
val y2 = tensor.matmul(tensor.mT)

//val y3 = torch.randLike(y1)
//torch.matmul(tensor, tensor.mT, out=y3)

// This computes the element-wise product. z1, z2, z3 will have the same value

val z1 = tensor * tensor
val z2 = tensor.mul(tensor)

//val z3 = torch.randLike(tensor)
//torch.mul(tensor, tensor, out=z3)
```

**Single-element tensors** If you have a one-element tensor, for example
by aggregating all values of a tensor into one value, you can convert it
to a Scala numerical value using `item()`:

```scala mdoc
val agg = tensor.sum
val aggItem = agg.item
print(aggItem)
println(aggItem.getClass)
```

**In-place operations** Operations that store the result into the
operand are called in-place. They are denoted by a `_` suffix. For
example: `x.copy_(y)`, `x.t_()`, will change `x`.

```scala mdoc
println(s"$tensor")
tensor -= 5
println(tensor)
```

@:callout(info)

In-place operations save some memory, but can be problematic when
computing derivatives because of an immediate loss of history. Hence,
their use is discouraged.

@:@