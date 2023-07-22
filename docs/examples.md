# Examples

You can find runnable examples in the [examples](https://github.com/sbrunk/storch/tree/main/examples/src/main/scala) directory.
We are planning to add more examples for different tasks.
If you have an idea or want to contribute or improve an example, please don't hesitate to open an issue or create a PR.

## Running the examples

You can clone the repository and run the examples as part of the build, for instance:

```bash
sbt> examples/runMain LeNetApp
```

The examples are also scala-cli scripts so you can run them with [scala-cli](https://scala-cli.virtuslab.org/), either locally or directly from the repo:

```bash
scala-cli https://raw.githubusercontent.com/sbrunk/storch/main/examples/src/main/scala/ImageClassifier.scala
```

## Image classifier example

Example script for training an image-classification model on your own images and running inference.
It uses the [ResNet](https://github.com/sbrunk/storch/blob/main/vision/src/main/scala/torchvision/models/resnet.scala) model implementation.

It will also automatically download converted pre-trained weights from the [releases](https://github.com/sbrunk/storch/releases/tag/pretrained-weights). See [converting pre-trained weights from PyTorch] for details.

### Training

To train a new image classifier on your own images run:

```bash
scala-cli ImageClassifier.scala -- train --dataset-dir <dataset>
```

Where the expected dataset is a directory per class with examples, like this:
```
.
├── PetImages
    ├── Cat
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   ├── ...
    └── Dog
        ├── 1.jpg
        ├── 2.jpg
        ├── ...
```

Using a smaller base model:
```bash
scala-cli ImageClassifier.scala -- train --base-model ResNet18 --dataset-dir <dataset>
```

To see all options, run:
```bash
scala-cli ImageClassifier.scala -- train -h
```

#### Training on the GPU

Right now, if you're using scala-cli you have to edit the directives at the top of the `ImageClassifier.scala`
script to enable GPU support (see comments in the script).
We're looking for ways to make this easier in the future.

### Inference

Once you've trained a model, you can use it for predicitons:
```bash
scala-cli ImageClassifier.scala -- predict --image-path <some-image.jpg>
```

If you don't have your own images, you can use an example dataset, for instance, the [Cat VS Dog dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset) (alternative [download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)) without requiring a kaggle account) is already in the right format.
