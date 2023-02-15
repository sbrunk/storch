/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import torch.{DType, Float32, Tensor, nn}
import torch.*
import torch.nn.functional as F
import torch.optim.SGD
import org.bytedeco.pytorch.OutputArchive
import torch.nn.modules.Default
import torchvision.datasets.MNIST
import scala.util.Random

class LeNet[D <: BFloat16 | Float32: Default] extends nn.Module {
  val conv1 = register(nn.Conv2d(1, 6, 5))
  val pool = register(nn.MaxPool2d((2, 2)))
  val conv2 = register(nn.Conv2d(6, 16, 5))
  val fc1 = register(nn.Linear(16 * 4 * 4, 120))
  val fc2 = register(nn.Linear(120, 84))
  val fc3 = register(nn.Linear(84, 10))

  def apply(i: Tensor[D]): Tensor[D] =
    var x = pool(F.relu(conv1(i)))
    x = pool(F.relu(conv2(x)))
    x = x.view(-1, 16 * 4 * 4) // all dimensions except the batch dimension
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = fc3(x)
    x
}

object LeNetApp extends App {
  val model = LeNet()

  // prepare data
  val path = "./data"
  val mnistTrain = MNIST(path, train = true)
  val mnistTest = MNIST("./data", train = false)
  val r = Random(seed = 0)
  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
    r.shuffle(mnistTrain).grouped(64).map { batch =>
      val (features, targets) = batch.unzip
      (torch.stack(features), torch.stack(targets))
    }

  val lossFn = torch.nn.loss.CrossEntropyLoss()
  val optimizer = SGD(model.parameters, lr = 0.01)

  // run training
  for (epoch <- 1 to 5) do
    for (batch <- dataLoader.zipWithIndex) do
      val ((feature, target), batchIndex) = batch
      optimizer.zeroGrad()
      val prediction = model(feature)
      val loss = lossFn(prediction, target)
      loss.backward()
      optimizer.step()
      if batchIndex % 100 == 0 then
        println("Epoch: " + epoch + " | Batch: " + f"$batchIndex%4d" + " | Loss: " + loss.item)
        // TODO eval
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to("net.pt")
}
