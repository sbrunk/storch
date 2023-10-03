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

class ReductionOpsSuite extends TensorCheckSuite {

  // TODO test with dim/keepdim variants for corresponding ops

  testUnaryOp(
    op = argmax(_),
    opName = "argmax",
    inputTensor =
      Tensor(Seq(1.0, 0.5, 1.2, -2)), // TODO check why we can't call tensor ops on inputTensor here
    expectedTensor = Tensor(2L)
  )

  testUnaryOp(
    op = argmin(_),
    opName = "argmin",
    inputTensor = Tensor(Seq(1.0, 0.5, 1.2, -2)),
    expectedTensor = Tensor(3L)
  )

  propertyTestUnaryOp(amax(_, 0), "amax")
  // TODO unit test amax

  propertyTestUnaryOp(amin(_, 0), "amin")
  // TODO unit test amin

  propertyTestUnaryOp(aminmax(_), "aminmax")
  // TODO unit test aminmax

  testUnaryOp(
    op = all,
    opName = "all",
    inputTensor = Tensor(Seq(true, true, false, true)),
    expectedTensor = Tensor(false)
  )

  test("all") {
    assertEquals(
      all(Tensor(Seq(true, true, false, true))),
      Tensor(false)
    )
    assertEquals(
      all(Tensor(Seq(true, true, true, true))),
      Tensor(true)
    )
  }

  testUnaryOp(
    op = any,
    opName = "any",
    inputTensor = Tensor(Seq(true, true, false, true)),
    expectedTensor = Tensor(true)
  )

  test("any") {
    assertEquals(
      any(Tensor(Seq(true, true, false, true))),
      Tensor(true)
    )
    assertEquals(
      any(Tensor(Seq(false, false, false, false))),
      Tensor(false)
    )
  }

  testUnaryOp(
    op = max,
    opName = "max",
    inputTensor = Tensor(Seq(1.0, 0.5, 1.2, -2)),
    expectedTensor = Tensor(1.2)
  )

  testUnaryOp(
    op = min,
    opName = "min",
    inputTensor = Tensor(Seq(1.0, 0.5, 1.2, -2)),
    expectedTensor = Tensor(-2.0)
  )

  // TODO Enable property test once we figure out to compile properly with AtLeastOneFloatOrComplex
  // propertyTestBinaryOp(dist35, "dist")

  def unitTestDist(p: Float, expected: Float) = unitTestBinaryOp[Float32, Float, Float32, Float](
    dist(_, _, p),
    "dist",
    inputTensors = (
      Tensor(Seq[Float](-1.5393, -0.8675, 0.5916, 1.6321)),
      Tensor(Seq[Float](0.0967, -1.0511, 0.6295, 0.8360))
    ),
    expectedTensor = Tensor(expected)
  )

  unitTestDist(3.5, 1.6727)
  unitTestDist(3, 1.6973)
  unitTestDist(0, 4)
  unitTestDist(1, 2.6537)

  propertyTestUnaryOp(logsumexp(_, dim = 0), "logsumexp")
  // TODO unit test logsumexp

  testUnaryOp(
    op = mean(_),
    opName = "mean",
    inputTensor = Tensor(Seq(0.2294, -0.5481, 1.3288)),
    expectedTensor = Tensor(0.3367)
  )

  test("mean with nan") {
    assert(mean(Tensor(Seq(Float.NaN, 1, 2))).isnan.item)
  }

  testUnaryOp(
    op = nanmean(_),
    opName = "nanmean",
    inputTensor = Tensor(Seq(Float.NaN, 1, 2, 1, 2, 3)),
    expectedTensor = Tensor(1.8f)
  )

  test("nanmean with nan") {
    val t = Tensor(Seq(Float.NaN, 1, 2))
    assert(!nanmean(t).isnan.item)
  }

  testUnaryOp(
    op = median,
    opName = "median",
    inputTensor = Tensor(Seq(1, 5, 2, 3, 4)),
    expectedTensor = Tensor(3)
  )

  testUnaryOp(
    op = nanmedian,
    opName = "nanmedian",
    inputTensor = Tensor(Seq(1, 5, Float.NaN, 3, 4)),
    expectedTensor = Tensor(3f)
  )

  propertyTestUnaryOp(mode(_), "mode")

  test("mode") {
    torch.manualSeed(0)
    assertEquals(
      torch.mode(Tensor(Seq(6, 5, 1, 0, 2)), 0),
      TensorTuple(Tensor(0), Tensor(3L))
    )
  }

  // test("mode") {
  //   torch.manualSeed(0)
  //   val a = Tensor(Seq(6, 5, 1, 0, 2))
  //   val b = a + Tensor(Seq(-3 , -11, -6, -7, 4)).reshape(5,1)
  //   val values = Tensor(Seq(-5, -5, -5, -4, -8))
  //   val indices = Tensor(Seq(1, 1, 1, 1, 1)).long
  //   assertEquals(
  //     torch.mode(b, 0),
  //     TensorTuple(values, indices)
  //   )
  // }

  testUnaryOp(
    op = nansum(_),
    opName = "nansum",
    inputTensor = Tensor(Seq(1, 5, Float.NaN, 3, 4)),
    expectedTensor = Tensor(13f)
  )

  testUnaryOp(
    op = prod,
    opName = "prod",
    inputTensor = Tensor(Seq(5.0, 5.0)),
    expectedTensor = Tensor(25.0)
  )

  // TODO quantile
  // TODO nanquantile

  propertyTestUnaryOp(std(_), "std")

  unitTestUnaryOp[Float32, Float](
    op = std(_, dim = 1, keepdim = true),
    opName = "std",
    inputTensor = Tensor(
      Seq(
        Seq[Float](0.2035, 1.2959, 1.8101, -0.4644),
        Seq[Float](1.5027, -0.3270, 0.5905, 0.6538),
        Seq[Float](-1.5745, 1.3330, -0.5596, -0.6548),
        Seq[Float](0.1264, -0.5080, 1.6420, 0.1992)
      ).flatten
    ).reshape(4, 4),
    expectedTensor = Tensor(Seq[Float](1.0311, 0.7477, 1.2204, 0.9087)).reshape(4, 1)
  )

  propertyTestUnaryOp(stdMean(_), "stdMean")
  // TODO unit test std_mean

  testUnaryOp(
    op = sum,
    opName = "sum",
    inputTensor = Tensor(Seq(5.0, 5.0)),
    expectedTensor = Tensor(10.0)
  )

  // TODO unique
  // TODO unique_consecutive

  propertyTestUnaryOp(variance(_), "variance")

  unitTestUnaryOp[Float32, Float](
    op = variance(_, dim = 1, keepdim = true),
    opName = "variance",
    inputTensor = Tensor(
      Seq(
        Seq[Float](0.2035, 1.2959, 1.8101, -0.4644),
        Seq[Float](1.5027, -0.3270, 0.5905, 0.6538),
        Seq[Float](-1.5745, 1.3330, -0.5596, -0.6548),
        Seq[Float](0.1264, -0.5080, 1.6420, 0.1992)
      ).flatten
    ).reshape(4, 4),
    expectedTensor = Tensor(Seq[Float](1.0631, 0.5590, 1.4893, 0.8258)).reshape(4, 1)
  )

  propertyTestUnaryOp(varMean(_), "varMean")
  // TODO unit test var_mean

  testUnaryOp(
    op = countNonzero(_),
    opName = "countNonzero",
    inputTensor = Tensor(Seq(1, 0, 0, 1, 0)),
    expectedTensor = Tensor(2L)
  )

}
