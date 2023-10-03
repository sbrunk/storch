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

class EmbeddingSuite extends munit.FunSuite {

  test("Embedding") {
    {
      torch.manualSeed(0)
      val embedding = nn.Embedding(10, 3)
      // a batch of 2 samples of 4 indices each
      val input = torch.Tensor(Seq(Seq(1L, 2, 4, 5), Seq(4L, 3, 2, 9)))
      val output = embedding(input)
      val expectedOutput = Tensor(
        Seq(
          Seq(
            Seq(-0.4339f, 0.8487f, 0.6920f),
            Seq(-0.3160f, -2.1152f, 0.3223f),
            Seq(0.1198f, 1.2377f, -0.1435f),
            Seq(-0.1116f, -0.6136f, 0.0316f)
          ),
          Seq(
            Seq(0.1198f, 1.2377f, -0.1435f),
            Seq(-1.2633f, 0.3500f, 0.3081f),
            Seq(-0.3160f, -2.1152f, 0.3223f),
            Seq(0.0525f, 0.5229f, 2.3022f)
          )
        )
      )
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      // example with padding_idx
      val embedding = nn.Embedding(5, 3, paddingIdx = Some(0))
      embedding.weight = Tensor(
        Seq(
          Seq(0f, 0f, 0f),
          Seq(0.5684f, -1.0845f, -1.3986f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(-0.8567f, 1.1006f, -1.0712f)
        )
      )
      val input = torch.Tensor(Seq(Seq(0L, 2, 0, 4)))
      val output = embedding(input)

      val expectedOutput = Tensor(
        Seq(
          Seq(0f, 0f, 0f),
          Seq(0.4033f, 0.8380f, -0.7193f),
          Seq(0f, 0f, 0f),
          Seq(-0.8567f, 1.1006f, -1.0712f)
        )
      ).unsqueeze(0)
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      //  example of changing `pad` vector
      val paddingIdx = 0
      val embedding = nn.Embedding(3, 3, paddingIdx = Some(paddingIdx))
      noGrad {
        embedding.weight(Seq(paddingIdx)) = torch.ones(3)
      }
      val expectedOutput = Tensor(
        Seq(
          Seq(1f, 1f, 1f),
          Seq(0.5684f, -1.0845f, -1.3986f),
          Seq(0.4033f, 0.8380f, -0.7193f)
        )
      )
      assert(torch.allclose(embedding.weight, expectedOutput, atol = 1e-4))
    }
  }
}
