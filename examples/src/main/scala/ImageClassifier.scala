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
//> using lib "me.tongfei:progressbar:0.9.5"
//> using lib "com.github.alexarchambault::case-app:2.1.0-M24"
//> using lib "org.scala-lang.modules::scala-parallel-collections:1.0.4"
// replace with pytorch-platform-gpu if you have a CUDA capable GPU
//> using lib "org.bytedeco:pytorch-platform:2.1.2-1.5.10"
// enable for CUDA support
////> using lib "org.bytedeco:cuda-platform-redist:12.3-8.9-1.5.10"
// enable for native Apple Silicon support
// will not be needed with newer versions of pytorch-platform
////> using lib "org.bytedeco:pytorch:2.1.2-1.5.10,classifier=macosx-arm64"

import Commands.*
import ImageClassifier.{Prediction, predict, train}
import caseapp.*
import caseapp.core.argparser.{ArgParser, SimpleArgParser}
import caseapp.core.app.CommandsEntryPoint
import com.sksamuel.scrimage.ImmutableImage
import me.tongfei.progressbar.ProgressBarBuilder
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.pytorch.{InputArchive, OutputArchive}
import os.Path
import torch.*
import torch.Device.{CPU, CUDA}
import torch.optim.Adam
import torchvision.models.resnet.{ResNet, ResNetVariant}

import java.nio.file.Paths
import scala.collection.parallel.CollectionConverters.ImmutableSeqIsParallelizable
import scala.util.{Random, Try, Using}

// format: off
/** Example script for training an image-classification model on your own images.
  *
  * Usage: scala-cli ImageClassifier.scala -- train <dataset>
  *  
  * Where the expected dataset is a directory per class with examples, like this:
  * .
  * ├── PetImages
  *     ├── Cat
  *     │   ├── 1.jpg
  *     │   ├── 2.jpg
  *     │   ├── ...
  *     └── Dog
  *         ├── 1.jpg
  *         ├── 2.jpg
  *         ├── ...
  * 
  * To see all options run scala-cli ImageClassifier.scala -- -h
  */
//format: on
object ImageClassifier extends CommandsEntryPoint:

  val fileTypes = Seq("jpg", "png")

  case class Metrics(loss: Float, accuracy: Float)

  torch.manualSeed(0)
  val random = new Random(seed = 0)

  extension (number: Double) def format: String = "%1.5f".format(number)

  def train(options: TrainOptions): Unit =
    val device = if torch.cuda.isAvailable then CUDA else CPU
    println(s"Using device: $device")

    val datasetDir = os.Path(options.datasetDir, base = os.pwd)

    /** verify that we can read all images of the dataset */
    if os
        .walk(datasetDir)
        .filter(path => fileTypes.contains(path.ext))
        .par
        .map { path =>
          val readTry = Try(ImmutableImage.loader().fromPath(path.toNIO)).map(_ => ())
          readTry.failed.foreach(e => println(s"Could not read $path"))
          readTry
        }
        .exists(_.isFailure)
    then
      println("Could not read all images in the dataset. Stopping.")
      System.exit(1)
    val classes = os.list(datasetDir).filter(os.isDir).map(_.last).sorted
    val classIndices = classes.zipWithIndex.toMap
    println(s"Found ${classIndices.size} classes: ${classIndices.mkString("[", ", ", "]")}")
    val pathsPerLabel = classes.map { label =>
      label -> os
        .list(datasetDir / label)
        .filter(path => fileTypes.contains(path.ext))
    }.toMap
    val pathsWithLabel =
      pathsPerLabel.toSeq.flatMap((label, paths) => paths.map(path => path -> label))
    println(s"Found ${pathsWithLabel.size} examples")
    println(
      pathsPerLabel
        .map((label, paths) => s"Found ${paths.size} examples for class $label")
        .mkString("\n")
    )

    val sample = random.shuffle(pathsWithLabel).take(options.take.getOrElse(pathsWithLabel.length))
    val (trainData, testData) = sample.splitAt((sample.size * 0.9).toInt)
    println(s"Train size: ${trainData.size}")
    println(s"Eval size:  ${testData.size}")

    val model: ResNet[Float32] = options.baseModel.factory(numClasses = classes.length)
    println(s"Model architecture: ${options.baseModel}")
    val transforms = options.baseModel.factory.DEFAULT.transforms

    if options.pretrained then
      val weights = torch.hub.loadStateDictFromUrl(options.baseModel.factory.DEFAULT.url)
      // Don't load the classification head weights, as we they are specific to the imagenet classes
      // and their output size (1000) usually won't match the number of classes of our dataset.
      model.loadStateDict(
        weights.filterNot((k, _) => Set("fc.weight", "fc.bias").contains(k))
      )
    model.to(device)

    val optimizer = Adam(model.parameters, lr = options.learningRate)
    val lossFn = torch.nn.loss.CrossEntropyLoss()
    val numEpochs = options.epochs
    val batchSize = options.batchSize
    val trainSteps = (trainData.size / batchSize.toFloat).ceil.toInt
    val evalSteps = (testData.size / options.batchSize.toFloat).ceil.toInt

    // Lazily loads inputs and transforms them into batches of tensors in the shape the model expects.
    def dataLoader(
        dataset: Seq[(Path, String)],
        shuffle: Boolean,
        batchSize: Int
    ): Iterator[(Tensor[Float32], Tensor[Int64])] =
      val loader = ImmutableImage.loader()
      (if shuffle then random.shuffle(dataset) else dataset)
        .grouped(batchSize)
        .map { batch =>
          val (inputs, labels) = batch.unzip
          // parallelize loading to improve GPU utilization
          val transformedInputs =
            inputs.par.map(path => transforms.transforms(loader.fromPath(path.toNIO))).seq
          assert(transformedInputs.forall(t => !t.isnan.any.item))
          (
            transforms.batchTransforms(torch.stack(transformedInputs)),
            torch.stack(labels.map(label => Tensor(classIndices(label)).to(dtype = int64)))
          )
        }

    def trainDL = dataLoader(trainData, shuffle = true, batchSize)

    def evaluate(): Metrics =
      val testDL = dataLoader(testData, shuffle = false, batchSize = batchSize)
      val evalPB =
        ProgressBarBuilder().setTaskName(s"Evaluating        ").setInitialMax(evalSteps).build()
      evalPB.setExtraMessage(s" " * 36)
      val isTraining = model.isTraining
      if isTraining then model.eval()
      val (loss, correct) = testDL
        .map { (inputBatch, labelBatch) =>
          // make sure we deallocate intermediate tensors in time
          Using.resource(new PointerScope()) { p =>
            val pred = model(inputBatch.to(device))
            val label = labelBatch.to(device)
            val loss = lossFn(pred, label).item
            val correct = pred.argmax(dim = 1).eq(label).sum.item
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

    for epoch <- 1 to numEpochs do
      val trainPB = ProgressBarBuilder()
        .setTaskName(s"Training epoch $epoch/$numEpochs")
        .setInitialMax(trainSteps)
        .build()
      var runningLoss = 0.0
      var step = 0
      var evalMetrics: Metrics = Metrics(Float.NaN, accuracy = 0)
      for (input, label) <- trainDL do {
        optimizer.zeroGrad()
        // Using PointerScope ensures that all intermediate tensors are deallocated in time
        Using.resource(new PointerScope()) { p =>
          val pred = model(input.to(device))
          val loss = lossFn(pred, label.to(device))
          loss.backward()
          // add a few sanity checks
          assert(
            model.parameters.forall(p => !p.isnan.any.item),
            "Parameters containing nan values"
          )
          assert(
            model.parameters.forall(p => !p.grad.exists(g => g.isnan.any.item)),
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
      }
      trainPB.close()
      println(
        s"Epoch $epoch/$numEpochs, Training loss: ${(runningLoss / step).format}, Evaluation loss: ${evalMetrics.loss.format}, Accuracy: ${evalMetrics.accuracy.format}"
      )
      val checkpointDir = os.Path(options.checkpointDir, os.pwd) / "%02d".format(epoch)
      os.makeDir.all(checkpointDir)
      val oa = OutputArchive()
      model.to(CPU).save(oa)
      oa.save_to((checkpointDir / "model.pt").toString)
      os.write(checkpointDir / "model.txt", options.baseModel.toString())
      os.write(checkpointDir / "classes.txt", classes.mkString("\n"))

  case class Prediction(label: String, confidence: Double)

  def predict(options: PredictOptions): Prediction =
    val modelDir = options.modelDir match
      case None =>
        val checkpoints = os.list(os.pwd / "checkpoints", sort = true)
        if checkpoints.isEmpty then
          println("Not checkpoint found. Did you train a model?")
          System.exit(1)
        checkpoints.last
      case Some(value) => os.Path(value, os.pwd)
    println(s"Trying to load model from $modelDir")
    val classes = os.read.lines(modelDir / "classes.txt")
    val modelVariant =
      ResNetVariant.valueOf(os.read(modelDir / "model.txt"))
    val model: ResNet[Float32] = modelVariant.factory(numClasses = classes.length)
    val transforms = modelVariant.factory.DEFAULT.transforms
    val ia = InputArchive()
    ia.load_from((modelDir / "model.pt").toString)
    model.load(ia)
    model.eval()
    val image = ImmutableImage.loader().fromPath(Paths.get(options.imagePath))
    val transformedImage =
      transforms.batchTransforms(transforms.transforms(image)).unsqueeze(dim = 0)
    val prediction = model(transformedImage)
    val TensorTuple(confidence, index) =
      torch.nn.functional.softmax(prediction, dim = 1)().max(dim = 1)
    val predictedLabel = classes(index.item.toInt)
    Prediction(predictedLabel, confidence.item)

  override def commands: Seq[Command[?]] = Seq(Train, Predict)
  override def progName: String = "image-classifier"

implicit val customArgParser: ArgParser[ResNetVariant] =
  SimpleArgParser.string.xmap(_.toString(), ResNetVariant.valueOf)

@HelpMessage("Train an image classification model")
case class TrainOptions(
    @HelpMessage(
      "Path to images. Images are expected to be stored in one directory per class i.e. cats/cat1.jpg cats/cat2.jpg dogs/dog1.jpg ..."
    )
    datasetDir: String,
    @HelpMessage(
      s"ResNet variant to use. Possible values are: ${ResNetVariant.values.mkString(", ")}. Defaults to ResNet50."
    )
    baseModel: ResNetVariant = ResNetVariant.ResNet50,
    @HelpMessage("Load pre-trained weights for base-model")
    pretrained: Boolean = true,
    @HelpMessage("Where to save model checkpoints")
    checkpointDir: String = "checkpoints",
    @HelpMessage("The maximum number of images to take for training")
    take: Option[Int] = None,
    batchSize: Int = 8,
    @HelpMessage("How many epochs (iterations over the input data) to train")
    epochs: Int = 1,
    learningRate: Double = 1e-5
)
@HelpMessage("Predict which class an image belongs to")
case class PredictOptions(
    @HelpMessage("Path to an image whose class we want to predict")
    imagePath: String,
    @HelpMessage(
      "Path to to the serialized model created by running 'train'. Tries the latest model in 'checkpoints' if not set."
    )
    modelDir: Option[String] = None
)

object Commands:
  object Train extends Command[TrainOptions]:
    override def run(options: TrainOptions, remainingArgs: RemainingArgs): Unit = train(options)
  object Predict extends Command[PredictOptions]:
    override def run(options: PredictOptions, remainingArgs: RemainingArgs): Unit =
      val Prediction(label, confidence) = predict(options)
      println(s"Class: $label, confidence: $confidence")
