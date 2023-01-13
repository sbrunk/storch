# Intro

```scala mdoc
val data = Seq(0,1,2,3)
val t1 = torch.Tensor(data)
t1.equal(torch.arange(0,4))
val t2 = t1.to(dtype=torch.float32)
val t3 = t1 + t2

val shape = Seq(2l,3l)
val randTensor = torch.rand(shape)
val zerosTensor = torch.zeros(shape, dtype=torch.int64)

val x = torch.ones(Seq(5))
val w = torch.randn(Seq(5, 3), requiresGrad=true)
val b = torch.randn(Seq(3), requiresGrad=true)
val z = (x matmul w) + b
```