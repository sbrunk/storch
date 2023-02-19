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

/** The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
  *
  * @param root
  *   Root directory of dataset where `train-images-idx3-ubyte` `t10k-images-idx3-ubyte` exist.
  * @param train
  *   If true, creates dataset from `train-images-idx3-ubyte`, otherwise from
  *   `t10k-images-idx3-ubyte`.
  */
class MNIST(root: Path, train: Boolean = true, download: Boolean = false)
    extends TensorDataset[Float32, Int64] {

  val mirrors = List(
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/"
  )

  val resources = List(
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
  )

  if download then doDownload()

  private val mode =
    if train then pytorch.MNIST.Mode.kTrain.intern().value
    else pytorch.MNIST.Mode.kTest.intern().value

  private val native = pytorch.MNIST(root.toString(), mode)

  private val ds =
    TensorDataset(Tensor[Float32](native.images().clone()), Tensor[Int64](native.targets().clone()))
  export ds.{apply, length, features, targets}

  private def downloadAndExtractArchive(url: URL, target: Path): Unit =
    println(s"$target not found, downloading from $url")
    Using.resource(url.openStream()) { inputStream =>
      Files.copy(GZIPInputStream(inputStream), target)
    }

  private def doDownload(): Unit =
    Files.createDirectories(root)
    for (filename, md5) <- resources do
      val finalPath = root.resolve(filename.stripSuffix(".gz"))
      if !Files.exists(finalPath) then
        downloadAndExtractArchive(new URL(s"${mirrors.head}$filename"), finalPath)

  override def toString(): String = ds.toString()
}
