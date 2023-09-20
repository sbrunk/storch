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

class ActivationSuite extends munit.FunSuite {
  test("LogSoftmax") {
    torch.manualSeed(0)
    val m = nn.LogSoftmax(dim = 1)
    val input = torch.randn(Seq(2, 3))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(
      Seq(
        Seq(-0.1689f, -2.0033f, -3.8886f),
        Seq(-0.2862f, -1.9392f, -2.2532f)
      )
    )
    assert(torch.allclose(output, expectedOutput, atol = 1e-4))
  }

  // TODO ReLU
  // TODO Softmax

  test("Tanh") {
    torch.manualSeed(0)
    val m = nn.Tanh()
    val input = torch.randn(Seq(2))
    val output = m(input)
    assertEquals(output.shape, input.shape)
    val expectedOutput = Tensor(Seq(0.9123f, -0.2853f))
    assert(torch.allclose(output, expectedOutput, atol = 1e-4))
  }

}
