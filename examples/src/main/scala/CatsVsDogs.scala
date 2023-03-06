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
//> using lib "dev.storch::vision:0.0-2ea690f-SNAPSHOT"
//> using lib "org.bytedeco:pytorch-platform:1.13.1-1.5.9-SNAPSHOT"
//> using lib "com.lihaoyi::os-lib:0.9.0"
//> using lib "me.tongfei:progressbar:0.9.5"

import com.sksamuel.scrimage.{ImmutableImage, ScaleMethod}
import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.pytorch.OutputArchive
import os.Path
import torch.Device.{CPU, CUDA}
import torch.{Float32, Int32, int64, Int64, Tensor}
import torch.optim.Adam
import torchvision.models.resnet.{ResNet18Weights, resnet18}

import java.nio.file.Paths
import scala.util.{Random, Using}
import scala.util.Try
import org.bytedeco.pytorch.InputArchive

object CatsVsDogs:

  torch.manualSeed(0)
  val random = new Random(seed = 0)

  val device = if torch.cuda.isAvailable then CUDA else CPU

  val weights = ResNet18Weights.IMAGENET1K_V1
  val model = resnet18(numClasses = 2)
  val lossFn = torch.nn.loss.CrossEntropyLoss()

  extension (number: Double) def format: String = "%1.5f".format(number)

  def dataloader(
      dataset: Seq[(Path, String)],
      shuffle: Boolean,
      batchSize: Int
  ): Iterator[(Tensor[Float32], Tensor[Int64])] =
    (if shuffle then random.shuffle(dataset) else dataset)
      .grouped(batchSize)
      .map { batch =>
        val (inputs, labels) = batch.unzip
        val transformedInputs =
          val loader = ImmutableImage.loader()
          inputs
            .map(path => loader.fromPath(path.toNIO))
            .map(weights.transforms.transforms)
        assert(transformedInputs.forall(t => !t.isnan.any.item))
        (
          weights.transforms.batchTransforms(
            torch.stack(transformedInputs).to(device)
          ),
          torch
            .stack(labels.map(label => Tensor(classIndices(label)).to(dtype = int64)))
            .to(device)
        )
      }

  val datasetDir = os.pwd / "data" / "PetImages"

  val classes = os.list(datasetDir).filter(os.isDir).map(_.last).sorted
  val classIndices = classes.zipWithIndex.toMap
  println(s"Found ${classIndices.size} classes: ${classIndices.mkString("[", ", ", "]")}")
  val pathsWithLabel = classes.flatMap(label =>
    (os
      .list(datasetDir / label)
      .filter(_.ext == "jpg")
      .map(path => path -> label))
  )

  println(s"Found ${pathsWithLabel.size} examples")

  val sample = random.shuffle(pathsWithLabel).take(1000)
  val (trainData, testData) = sample.splitAt((sample.size * 0.9).toInt)
  println(s"Train size: ${trainData.size}")
  println(s"Eval size:  ${testData.size}")

  val evalBatchSize = 16
  def testDL = dataloader(testData, shuffle = false, batchSize = evalBatchSize)
  val evalSteps = (testData.size / evalBatchSize.toFloat).ceil.toInt

  def train(): Unit =
    val weights = torch.pickleLoad(Paths.get("notebooks/resnet18.pt"))
    model.loadStateDict(
      weights.filterNot((k, v) => Set("fc.weight", "fc.bias").contains(k))
    )
    model.to(device)
    val optimizer = Adam(model.parameters, lr = 1e-5)
    val numEpochs = 1
    val batchSize = 16
    val trainSteps = (trainData.size / batchSize.toFloat).ceil.toInt

    def trainDL = dataloader(trainData, shuffle = true, batchSize)

    for epoch <- 1 to numEpochs do
      var trainPB = ProgressBarBuilder()
        .setTaskName(s"Training epoch $epoch/$numEpochs")
        .setInitialMax(trainSteps)
        .build()
      var runningLoss = 0.0
      var step = 0
      var evalMetrics: Metrics = Metrics(Float.NaN, accuracy = 0)
      for (input, label) <- trainDL do {
        optimizer.zeroGrad()
        // PointerScope makes sure that all intermediate tensors are deallocated in time
        Using.resource(new PointerScope()) { p =>
          val pred = model(input.to(device))
          val loss = lossFn(pred, label.to(device))
          loss.backward()
          // add a few sanity checks
          assert(
            model.parameters.forall(t => !t.isnan.any.item),
            "Parameters containing nan values"
          )
          assert(
            model.parameters.forall(t => !t.grad.isnan.any.item),
            "Gradients containing nan values"
          )
          optimizer.step()
          runningLoss += loss.item
        }
        trainPB.setExtraMessage(" " * 21 + s"Loss: ${(runningLoss / step).format}")
        trainPB.step()
        if ((step + 1) % (trainSteps / 4.0)).toInt == 0 then
          evalMetrics = evaluate()
          runningLoss = 0.0
        step += 1
        step
      }
      trainPB.close()
      println(
        s"Epoch $epoch/$numEpochs, Training loss: ${(runningLoss / step).format}, Evaluation loss: ${evalMetrics.loss.format}, Accuracy: ${evalMetrics.accuracy.format}"
      )
      val oa = OutputArchive()
      model.save(oa)
      oa.save_to(s"checkpoint-$epoch.pt")

  case class Metrics(loss: Float, accuracy: Float)

  def evaluate(): Metrics =
    val evalPB =
      ProgressBarBuilder().setTaskName(s"Evaluating        ").setInitialMax(evalSteps).build()
    evalPB.setExtraMessage(s" " * 36)
    val isTraining = model.isTraining
    if isTraining then model.eval()
    val (loss, correct) = testDL
      .map { (inputBatch, labelBatch) =>
        Using.resource(new PointerScope()) { p =>
          var pred = model(inputBatch.to(device))
          var loss = lossFn(pred, labelBatch.to(device)).item
          var correct = pred.argmax(dim = 1).eq(labelBatch).sum.item
          evalPB.step()
          (loss, correct)
        }
      }
      .toSeq
      .unzip
    val metrics = Metrics(
      Tensor(loss).mean.item,
      Tensor(correct).sum.item / testData.size.toFloat
    )
    evalPB.setExtraMessage(
      s"    Loss: ${metrics.loss.format}, Accuracy: ${metrics.accuracy.format}"
    )
    evalPB.close()
    if isTraining then model.train()
    metrics

  def cleanup(): Unit =
    os.walk(datasetDir).filter(_.ext == "jpg").foreach { path =>
      Try(ImmutableImage.loader().fromPath(path.toNIO)).recover { _ =>
        println(s"Cleaning up broken image $path")
        os.move(path, path / os.up / (path.last + ".bak"))
      }
    }

  def predict(imagePath: String) =
    val ia = InputArchive()
    ia.load_from("checkpoint-2.pt")
    model.load(ia)
    model.eval()
    val image = ImmutableImage.loader().fromPath(Paths.get(imagePath))
    val transformedImage =
      weights.transforms.batchTransforms(weights.transforms.transforms(image)).unsqueeze(dim = 0)
    val prediction = model(transformedImage)
    val correct = prediction.argmax(dim = 1).item

  def main(args: Array[String]): Unit =
    // train()
    predict(args(0))
