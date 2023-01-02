# Storch - GPU Accelerated Deep Learning for Scala 3

Storch is a Scala library for fast tensor computations and deep learning, based on [PyTorch](https://pytorch.org/).

Like PyTorch, Storch provides
* A NumPy like API for working with tensors
* GPU support
* Automatic differentiation
* A neural network API for building and training neural networks.

Storch aims to close to the Python API to make porting existing models and the life of people already familiar with PyTorch easier.

```scala
val data = Seq(0,1,2,3)
// data: Seq[Int] = List(0, 1, 2, 3)
val t1 = torch.Tensor(data)
// t1: Tensor[Int32] = dtype=int32, shape=[4], device=CPU 
// [0, 1, 2, 3]
t1.equal(torch.arange(0,4))
// res0: Boolean = true
val t2 = t1.to(dtype=float32)
// t2: Tensor[Float32] = dtype=float32, shape=[4], device=CPU 
// [0,0000, 1,0000, 2,0000, 3,0000]
val t3 = t1 + t2
// t3: Tensor[Float32] = dtype=float32, shape=[4], device=CPU 
// [0,0000, 2,0000, 4,0000, 6,0000]

val shape = Seq(2l,3l)
// shape: Seq[Long] = List(2, 3)
val randTensor = torch.rand(shape)
// randTensor: Tensor[Float32] = dtype=float32, shape=[2, 3], device=CPU 
// [[0,4341, 0,9738, 0,9305],
//  [0,8987, 0,1122, 0,3912]]
val zerosTensor = torch.zeros(shape, dtype=torch.int64)
// zerosTensor: Tensor[Int64] = dtype=int64, shape=[2, 3], device=CPU 
// [[0, 0, 0],
//  [0, 0, 0]]

val x = torch.ones(Seq(5))
// x: Tensor[Float32] = dtype=float32, shape=[5], device=CPU 
// [1,0000, 1,0000, 1,0000, 1,0000, 1,0000]
val w = torch.randn(Seq(5, 3), requiresGrad=true)
// w: Tensor[Float32] = dtype=float32, shape=[5, 3], device=CPU 
// [[0,8975, 0,5484, 0,2307],
//  [0,2689, 0,7430, 0,6446],
//  [0,9503, 0,6342, 0,7523],
//  [0,5332, 0,7497, 0,3665],
//  [0,3376, 0,6040, 0,5033]]
val b = torch.randn(Seq(3), requiresGrad=true)
// b: Tensor[Float32] = dtype=float32, shape=[3], device=CPU 
// [0,2638, 0,9697, 0,3664]
val z = (x matmul w) + b
// z: Tensor[Float32] = dtype=float32, shape=[3], device=CPU 
// [3,2513, 4,2490, 2,8640]
```

One notable difference is that tensors in Storch are statically typed regarding the underlying `dtype`.
So you'll see `Tensor[Float32]` or `Tensor[Int8]` instead of just `Tensor`.

Tracking the data type at compile time enables us to catch certain errors earlier. For instance, `torch.rand` is only implemented for float types and the following will trigger a runtime error in PyTorch:
```python
torch.rand([3,3], dtype=torch.int32) # RuntimeError: "check_uniform_bounds" not implemented for 'Int'
```

In Storch, the same code does not compile:
```scala
torch.rand(Seq(3,3), dtype=torch.int32) // compile error
```

Example module:
```scala
class LeNet[D <: BFloat16 | Float32] extends nn.Module:
  val conv1 = register(nn.Conv2d[D](1, 6, 5))
  val pool  = nn.MaxPool2d[D]((2, 2))
  val conv2 = register(nn.Conv2d[D](6, 16, 5))
  val fc1   = register(nn.Linear[D](16 * 4 * 4, 120))
  val fc2   = register(nn.Linear[D](120, 84))
  val fc3   = register(nn.Linear[D](84, 10))

  def apply(i: Tensor[D]): Tensor[D] =
    var x = pool(F.relu(conv1(i)))
    x = pool(F.relu(conv2(x)))
    x = x.view(-1, 16 * 4 * 4)
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = fc3(x)
    x
```

Storch is powered by [LibTorch](https://pytorch.org/cppdocs/index.html), the C++ library underlying PyTorch and [JavaCPP](https://github.com/bytedeco/javacpp), which provides generated Java bindings for LibTorch as well as important utilities to integrate with native code on the JVM.

## Installation

As Storch is still in an early stage of development, there are no published artifacts available yet, so you'll have to build Storch from source.

### Building from source

Storch uses [Bleep](https://bleep.build/docs/) as build tool. The build is defined in `bleep.yaml`.
Don't worry if you are not familiar with Bleep, it's refreshingly simple and easy to use.

1. Install [Coursier](https://get-coursier.io/docs/cli-installation)
2. Install Bleep
   ```bash
   cs install --channel https://raw.githubusercontent.com/oyvindberg/bleep/master/coursier-channel.json bleep
   ```
3. Clone the Storch repo
4. Build Storch and publish it locally
   ```bash
   bleep publish-local
   ```

Now you're ready to use Storch.
