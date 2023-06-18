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

class RandomSamplingOpsSuite extends TensorCheckSuite {

  testUnaryOp(
    op = multinomial(_, 2, true),
    opName = "multinomial",
    inputTensor = Tensor(Seq(0.0, 0.0, 0.0, 1.0)),
    expectedTensor = Tensor(Seq(3L, 3L))
  )

  test("randint.unit-test") {
    // randint generates uniform numbers in the range [min, max)
    val low = 0
    val high = 4
    val randintTensor = randint(low, high + 1, Seq(100000)).to(dtype = float32)
    val randintMean = randintTensor.mean
    val expectedMean = Tensor(high / 2).to(dtype = float32)

    println(randintMean)
    println(expectedMean)
    assert(allclose(randintMean, expectedMean, atol = 1e-2))
  }

  test("randn.unit-test") {
    val randnTensor = randn(Seq(100000))
    val randnMean = randnTensor.mean
    val expectedMean = Tensor(0.0).to(dtype = float32)
    val randnVariance = randnTensor.variance
    val expectedVariance = Tensor(1.0).to(dtype = float32)

    assert(
      allclose(randnMean, expectedMean, atol = 1e-2) &&
        allclose(randnVariance, expectedVariance, atol = 1e-2)
    )
  }

}
