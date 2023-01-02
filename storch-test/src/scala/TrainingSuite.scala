package torch
import torch.data.*

class TraininSuite extends munit.FunSuite {
  test("training") {

    val xTrain     = torch.arange(end = 10, dtype = float32) // .reshape(10, 1)
    val yTrain     = Tensor(1.0f, 1.3f, 3.1f, 2.0f, 5.0f, 6.3f, 6.6f, 7.4f, 8.0f, 9.0f)
    val xTrainNorm = ((xTrain - xTrain.mean) / xTrain.std)

    val ds = TensorSeq(xTrainNorm).zip(TensorSeq(yTrain))

    torch.manualSeed(1)

    var weight = torch.randn(Seq(1), requiresGrad = true)
    var bias   = torch.zeros(Seq(1), requiresGrad = true)

    def model(xb: Tensor[Float32]): Tensor[Float32] = (xb matmul weight) + bias

    def lossFn(input:  Tensor[Float32], target:  Tensor[Float32]) = (input - target).pow(2).mean

    val learningRate = 0.001f
    val numEpochs    = 10
    val logEpochs    = 1

    Range(0, xTrainNorm.size.head.toInt, 2)

    val dl = TupleDataLoader(ds, batchSize = 1, shuffle = true)

    val batch = dl.head
    val pred  = model(batch._1)
    val loss  = lossFn(pred, batch._2)
    loss.backward()

    for {
      epoch <- 0 to numEpochs
      loss = dl.map { (x, y) =>
        val pred = model(x)
        val loss = lossFn(pred, y)
        loss.backward()
        noGrad {
          weight -= weight.grad * learningRate
          bias -= bias.grad * learningRate
          weight.grad.zero()
          bias.grad.zero()
        }
        loss
      }.last
    } {
      if (epoch % logEpochs == 0) println(s"Epoch ${epoch} Loss ${loss.item}")
    }
  }
}
