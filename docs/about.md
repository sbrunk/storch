# About

Storch is a Scala library for fast tensor computations and deep learning, based on [PyTorch](https://pytorch.org/).

Like PyTorch, Storch provides

  * A NumPy like API for working with tensors
  * GPU support
  * Automatic differentiation
  * A neural network API for building and training neural networks.

Storch aims to stay close to the original PyTorch API to make porting existing models and the life of people already familiar with PyTorch easier.

```scala mdoc:invisible
torch.manualSeed(0)
```

```scala mdoc
val data = Seq(0,1,2,3)
val t1 = torch.Tensor(data)
t1.equal(torch.arange(0,4))
val t2 = t1.to(dtype=torch.float32)
val t3 = t1 + t2

val shape = Seq(2,3)
val randTensor = torch.rand(shape)
val zerosTensor = torch.zeros(shape, dtype=torch.int64)

val x = torch.ones(Seq(5))
val w = torch.randn(Seq(5, 3), requiresGrad=true)
val b = torch.randn(Seq(3), requiresGrad=true)
val z = (x matmul w) + b
```

One notable difference is that tensors in Storch are statically typed regarding the underlying `dtype`.
So you'll see `Tensor[Float32]` or `Tensor[Int8]` instead of just `Tensor`.

Tracking the data type at compile time enables us to catch certain errors earlier. For instance, `torch.rand` is only implemented for float types and the following will trigger a runtime error in PyTorch:
```python
torch.rand([3,3], dtype=torch.int32) # RuntimeError: "check_uniform_bounds" not implemented for 'Int'
```

In Storch, the same code does not compile:
```scala mdoc:fail
torch.rand(Seq(3,3), dtype=torch.int32)
```

Storch is powered by [LibTorch](https://pytorch.org/cppdocs/index.html), the C++ library underlying PyTorch and
[JavaCPP](https://github.com/bytedeco/javacpp), which provides generated Java bindings for LibTorch as well as important utilities to integrate with native code on the JVM.