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

//> using scala "3.2"
//> using repository "sonatype-s01:snapshots"
//> using repository "sonatype:snapshots"
//> using lib "dev.storch::vision:0.0-2b6ed09-SNAPSHOT"
//> using lib "org.bytedeco:pytorch-platform:1.13.1-1.5.9-SNAPSHOT"

import torch.*
import torch.nn.functional as F
import torch.optim.Adam
import org.bytedeco.pytorch.OutputArchive
import torch.nn.modules.Default
import torchvision.datasets.MNIST
import scala.util.Random
import java.nio.file.Paths

// define model architecture
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
  val dataPath = Paths.get("data")
  val mnistTrain = MNIST(dataPath, train = true, download = true)
  val mnistEval = MNIST(dataPath, train = false)
  val r = Random(seed = 0)
  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
    r.shuffle(mnistTrain).grouped(32).map { batch =>
      val (features, targets) = batch.unzip
      (torch.stack(features), torch.stack(targets))
    }

  val lossFn = torch.nn.loss.CrossEntropyLoss()
  val optimizer = Adam(model.parameters, lr = 0.001)

  // run training
  for (epoch <- 1 to 5) do
    for (batch <- dataLoader.zipWithIndex) do
      val ((feature, target), batchIndex) = batch
      optimizer.zeroGrad()
      val prediction = model(feature)
      val loss = lossFn(prediction, target)
      loss.backward()
      optimizer.step()
      if batchIndex % 200 == 0 then
        // run evaluation
        val predictions = model(mnistEval.features)
        val evalLoss = lossFn(predictions, mnistEval.targets)
        val accuracy =
          (predictions.argmax(dim = 1).eq(mnistEval.targets).sum / mnistEval.length).item
        println(
          f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
        )
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to("net.pt")
}
