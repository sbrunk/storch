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
package modules

class BatchNormSuite extends munit.FunSuite {

  test("BatchNorm1d") {
    torch.manualSeed(0)
    val m = nn.BatchNorm1d(numFeatures = 3)
    val input = torch.randn(Seq(3, 3))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(1.4014f, -0.1438f, -1.2519f),
        Seq(-0.5362f, -1.1465f, 0.0564f),
        Seq(-0.8651f, 1.2903f, 1.1956f)
      )
    )
    assert(torch.allclose(output, expectedOutput, atol = 1e-4))
  }

  test("BatchNorm2d") {
    torch.manualSeed(0)
    val m = nn.BatchNorm2d(numFeatures = 3)
    val input = torch.randn(Seq(3, 3, 1, 1))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(1.4014f, -0.1438f, -1.2519f),
        Seq(-0.5362f, -1.1465f, 0.0564f),
        Seq(-0.8651f, 1.2903f, 1.1956f)
      )
    )
    assert(torch.allclose(output.squeeze, expectedOutput, atol = 1e-4))
  }
}
