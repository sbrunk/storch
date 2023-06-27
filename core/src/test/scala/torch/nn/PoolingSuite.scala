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
import org.scalacheck.Gen
import torch.Generators.allDTypes

class PoolingSuite extends TensorCheckSuite {
  test("MaxPool2d output shapes") {
    val input = torch.randn(Seq(1, 3, 244, 244))
    // pool of square window of size=3, stride=2
    val m1 = MaxPool2d[Float32](3, stride = Some(2))
    assertEquals(m1(input).shape, Seq(1, 3, 121, 121))
    // pool of non-square window
    val m2 = MaxPool2d[Float32]((3, 2), stride = Some(2, 1))
    assertEquals(m2(input).shape, Seq(1, 3, 121, 243))
    val m3 = MaxPool2d[Float32](3)
    assertEquals(m3(input).shape, Seq(1, 3, 81, 81))
  }

  val shape3d = Seq(16, 50, 32)
  propertyTestUnaryOp(F.avgPool1d(_, 3), "avgPool1d", genRandTensor(shape3d))
  propertyTestUnaryOp(F.maxPool1d(_, 3), "maxPool1d", genRandTensor(shape3d))
  propertyTestUnaryOp(F.maxPool1dWithIndices(_, 3), "maxPool1dWithIndices", genRandTensor(shape3d))

  inline def genRandTensor[D <: FloatNN | ComplexNN](shape: Seq[Int] = Seq(3, 3)): Gen[Tensor[D]] =
    Gen.oneOf(allDTypes.filter(_.isInstanceOf[D])).map { dtype =>
      torch.rand(shape, dtype = dtype.asInstanceOf[D])
    }

  val shape4d = Seq(8, 16, 50, 32)
  propertyTestUnaryOp(F.avgPool2d(_, 3), "avgPool2d", genRandTensor(shape4d))
  propertyTestUnaryOp(F.maxPool2d(_, 3), "maxPool2d", genRandTensor(shape4d))
  propertyTestUnaryOp(F.maxPool2dWithIndices(_, 3), "maxPool2dWithIndices", genRandTensor(shape4d))

  val shape5d = Seq(2, 16, 50, 44, 31)
  propertyTestUnaryOp(
    F.avgPool3d(_, (3, 2, 2), stride = (2, 1, 2)),
    "avgPool3d",
    genRandTensor(shape5d)
  )
  propertyTestUnaryOp(
    F.maxPool3d(_, (3, 2, 2), stride = (2, 1, 2)),
    "maxPool3d",
    genRandTensor(shape5d)
  )
  propertyTestUnaryOp(
    F.maxPool3dWithIndices(_, (3, 2, 2), stride = (2, 1, 2)),
    "maxPool3dWithIndices",
    genRandTensor(shape5d)
  )
}
