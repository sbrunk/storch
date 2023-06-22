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

package torch
package nn
import functional as F

class ConvolutionSuite extends munit.FunSuite {

  test("mismatchShapeConv2d") {
    val dtypes = List[FloatNN | ComplexNN](torch.float32, torch.complex64)
    for (dtype <- dtypes) {
      val x = torch.randn(Seq(1, 10, 1, 28, 28), dtype)
      val w = torch.randn(Seq(6, 1, 5, 5), dtype)

      intercept[RuntimeException](F.conv2d(x, w))
      // TODO find a way to run interceptMessage comparing only the first line/string prefix as we don't care about the c++ stacktrace here
      //   interceptMessage[RuntimeException] {
      //     """Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 10, 1, 28, 28]"""
      //   } {
      //     conv2d(x, w)
      //   }
    }
  }

}
