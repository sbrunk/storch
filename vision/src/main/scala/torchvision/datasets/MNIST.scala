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

package torchvision.datasets

import torch.*
import org.bytedeco.pytorch
import torch.data.TensorDataset
import java.nio.file.Path
import scala.util.Using
import java.net.URL
import java.nio.file.Files
import java.util.zip.GZIPInputStream
import scala.util.Try
import scala.util.Success
import scala.util.Failure
import torch.Tensor.fromNative

trait MNISTBase(
    val mirrors: Seq[String],
    val resources: Seq[(String, String)],
    val classes: Seq[String],
    val root: Path,
    val train: Boolean,
    val download: Boolean
) extends TensorDataset[Float32, Int64] {

  private def downloadAndExtractArchive(url: URL, target: Path): Unit =
    println(s"downloading from $url")
    Using.resource(url.openStream()) { inputStream =>
      val _ = Files.copy(GZIPInputStream(inputStream), target)
    }

  if download then {
    Files.createDirectories(root)
    for (filename, md5) <- resources do
      val finalPath = root.resolve(filename.stripSuffix(".gz"))
      if !Files.exists(finalPath) then
        println(s"$finalPath not found")
        val _ = mirrors.iterator
          .map { mirror =>
            Try(downloadAndExtractArchive(URL(s"$mirror$filename"), finalPath))
          }
          .tapEach {
            case Failure(exception) => println(exception)
            case Success(_)         =>
          }
          .collectFirst { case Success(_) => }
  }

  private val mode =
    if train then pytorch.MNIST.Mode.kTrain.intern().value
    else pytorch.MNIST.Mode.kTest.intern().value

  private val native = pytorch.MNIST(root.toString(), mode)

  private val ds =
    TensorDataset(
      fromNative[Float32](native.images().clone()),
      fromNative[Int64](native.targets().clone())
    )
  export ds.{apply, length, features, targets}

  override def toString(): String = ds.toString()
}

/** The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
  *
  * @param root
  *   Root directory of dataset where `train-images-idx3-ubyte` `t10k-images-idx3-ubyte` exist.
  * @param train
  *   If true, creates dataset from `train-images-idx3-ubyte`, otherwise from
  *   `t10k-images-idx3-ubyte`.
  */
class MNIST(root: Path, train: Boolean = true, download: Boolean = false)
    extends MNISTBase(
      mirrors = List(
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
      ),
      resources = List(
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
      ),
      classes = Seq(
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine"
      ),
      root,
      train,
      download
    )

/** The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) Dataset.
  *
  * @param root
  *   Root directory of dataset where `train-images-idx3-ubyte` `t10k-images-idx3-ubyte` exist.
  * @param train
  *   If true, creates dataset from `train-images-idx3-ubyte`, otherwise from
  *   `t10k-images-idx3-ubyte`.
  */
class FashionMNIST(root: Path, train: Boolean = true, download: Boolean = false)
    extends MNISTBase(
      mirrors = List(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
      ),
      resources = List(
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
      ),
      classes = Seq(
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
      ),
      root,
      train,
      download
    )
