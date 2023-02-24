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

package torchvision
package datasets

import java.nio.file.Paths

class MNISTSuite extends munit.FunSuite {
  test("MNIST download") {
    val mnistTrain = MNIST(Paths.get("data/mnist"), download = true)
    assertEquals(mnistTrain.features.shape, Seq(60000, 1, 28, 28))
    val mnistTest = MNIST(Paths.get("data/mnist"), train = false, download = true)
    assertEquals(mnistTest.features.shape, Seq(10000, 1, 28, 28))
  }

  test("FashionMNIST download") {
    val fashionMNISTTrain = FashionMNIST(Paths.get("data/fashion-mnist"), download = true)
    assertEquals(fashionMNISTTrain.features.shape, Seq(60000, 1, 28, 28))
    val fashionMNISTTest =
      FashionMNIST(Paths.get("data/fashion-mnist"), train = false, download = true)
    assertEquals(fashionMNISTTest.features.shape, Seq(10000, 1, 28, 28))
  }

}
