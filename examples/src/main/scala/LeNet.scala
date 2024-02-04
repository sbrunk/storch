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

//> using scala "3.3"
//> using repository "sonatype:snapshots"
//> using repository "sonatype-s01:snapshots"
//> using lib "dev.storch::vision:0.0-2fff591-SNAPSHOT"
// replace with pytorch-platform-gpu if you have a CUDA capable GPU
//> using lib "org.bytedeco:pytorch-platform:2.1.2-1.5.10"
// enable for CUDA support
////> using lib "org.bytedeco:cuda-platform-redist:12.3-8.9-1.5.10"
// enable for native Apple Silicon support
// will not be needed with newer versions of pytorch-platform
////> using lib "org.bytedeco:pytorch:2.1.2-1.5.10,classifier=macosx-arm64"

import torch.*
import torch.nn.functional as F
import torch.optim.Adam
import org.bytedeco.pytorch.OutputArchive
import torchvision.datasets.MNIST
import scala.util.Random
import java.nio.file.Paths
import torch.Device.CUDA
import scala.util.Using
import org.bytedeco.javacpp.PointerScope
import torch.Device.CPU
import torch.nn.modules.HasParams

// Define the model architecture
class LeNet[D <: BFloat16 | Float32: Default] extends HasParams[D] {

  val conv1 = register(nn.Conv2d(1, 6, 5))
  val conv2 = register(nn.Conv2d(6, 16, 5))
  val fc1 = register(nn.Linear(16 * 4 * 4, 120))
  val fc2 = register(nn.Linear(120, 84))
  val fc3 = register(nn.Linear(84, 10))

  def apply(i: Tensor[D]): Tensor[D] =
    var x = F.maxPool2d(F.relu(conv1(i)), (2, 2))
    x = F.maxPool2d(F.relu(conv2(x)), 2)
    x = x.view(-1, 16 * 4 * 4) // all dimensions except the batch dimension
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = fc3(x)
    x
}

/** Shows how to train a simple LeNet on the MNIST dataset */
object LeNetApp extends App {
  val device = if torch.cuda.isAvailable then CUDA else CPU
  println(s"Using device: $device")
  val model = LeNet().to(device)

  // prepare data
  val dataPath = Paths.get("data/mnist")
  val mnistTrain = MNIST(dataPath, train = true, download = true)
  val mnistEval = MNIST(dataPath, train = false)
  val evalFeatures = mnistEval.features.to(device)
  val evalTargets = mnistEval.targets.to(device)
  val r = Random(seed = 0)

  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
    r.shuffle(mnistTrain).grouped(32).map { batch =>
      val (features, targets) = batch.unzip
      (torch.stack(features).to(device), torch.stack(targets).to(device))
    }

  val lossFn = torch.nn.loss.CrossEntropyLoss()
  // enable AMSGrad to avoid convergence issues
  val optimizer = Adam(model.parameters, lr = 1e-3, amsgrad = true)

  // run training
  for (epoch <- 1 to 5) do
    for (batch <- dataLoader.zipWithIndex) do
      // make sure we deallocate intermediate tensors in time
      Using.resource(new PointerScope()) { p =>
        val ((feature, target), batchIndex) = batch
        optimizer.zeroGrad()
        val prediction = model(feature)
        val loss = lossFn(prediction, target)
        loss.backward()
        optimizer.step()
        if batchIndex % 200 == 0 then
          // run evaluation
          val predictions = model(evalFeatures)
          val evalLoss = lossFn(predictions, evalTargets)
          val accuracy =
            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
          println(
            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
          )
      }
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to("net.pt")
}
