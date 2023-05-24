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

import DeviceType.CUDA

import java.nio.{IntBuffer, LongBuffer}

import munit.ScalaCheckSuite
import torch.DeviceType.CUDA
import org.scalacheck.Prop.*
import org.bytedeco.pytorch.global.torch as torch_native
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck._
import Gen._
import Arbitrary.arbitrary
import DeviceType.CPU
import Generators.{*, given}
import scala.util.Try
import spire.math.Complex
import spire.implicits.DoubleAlgebra

class TensorSuite extends ScalaCheckSuite {

  inline private def testUnaryOp[In <: DType, InS <: ScalaType](
      op: Tensor[In] => Tensor[?],
      opName: String,
      inline inputTensor: Tensor[ScalaToDType[InS]],
      inline expectedTensor: Tensor[?],
      absolutePrecision: Double = 1e-04
  )(using ScalaToDType[InS] <:< In): Unit =
    val propertyTestName = s"${opName}.property-test"
    test(propertyTestName) {
      forAll(genTensor[In]) { (tensor) =>
        val result = Try(op(tensor))
        // TODO Validate output types
        assert(
          result.isSuccess,
          s"""|
              |Tensor operation 'torch.${opName}' does not support ${tensor.dtype} inputs
              |
              |${result.failed.get}
              """.stripMargin
        )
      }
    }
    val unitTestName = s"${opName}.unit-test"
    test(unitTestName) {
      val outputTensor = op(inputTensor.asInstanceOf[Tensor[In]])
      val allclose = outputTensor.allclose(
        other = expectedTensor,
        atol = absolutePrecision,
        equalNan = true
      )
      assert(
        allclose,
        s"""|
            |Tensor results are not all close for 'torch.${opName}'
            |
            |Input tensor:
            |${inputTensor}
            |
            |Output tensor:
            |${outputTensor}
            |
            |Expected tensor:
            |${expectedTensor}""".stripMargin
      )
    }

  test("arange") {
    val t0 = arange(0, 10)
    assertEquals(t0.toSeq, Seq.range(0, 10))
    val t1 = arange(0, 10, 2)
    assertEquals(t1.toSeq, Seq.range(0, 10, 2))
  }

  test("tensor properties") {
    val t = ones(Seq(2, 3), dtype = float32)
    assertEquals(t.size, Seq[Int](2, 3))
    assertEquals(t.device, Device(DeviceType.CPU, -1: Byte))
    assertEquals(t.numel, 2L * 3)
  }

  property("tensor dtypes") {
    forAll { (dtype: DType) =>
      val t = ones(Seq(2, 3), dtype)
      assertEquals(t.dtype, dtype)
    }
  }

  // property("tensor requiresGrad") {
  //   forAll { (dtype: FloatNN | ComplexNN, requiresGrad: Boolean) =>
  //     val t = ones(Seq(2, 3), dtype, requiresGrad=requiresGrad)
  //     assertEquals(t.dtype, dtype)
  //   }
  // }

  property("tensor ones") {
    forAll(genTensorSize, genDType) { (size, dtype) =>
      val t = ones(size, dtype)
      assertEquals(t.dtype, dtype)
      assertEquals(t.size, size)
      assertEquals(t.numel, size.product.toLong)
      assertEquals(t.toSeq.length, size.product.toInt)
    }
  }

  test("ones") {
    val t = ones[Float32](Seq(2, 3))
    assertEquals(t.size, Seq(2, 3))
    assertEquals(t.numel, 2L * 3)
    assertEquals(t.toSeq, Seq.fill[Float](2 * 3)(1f))
  }

  test("exp and log") {
    val t = Tensor(Seq(1.0, 2.0, 3.0))
    assertEquals(t.log(0), Tensor(0.0))
    assert(torch.allclose(t.log.exp, t))
  }

  test("toBuffer") {
    val content = Seq(1, 2, 3, 4)
    val t = Tensor(content)
    val b = t.toBuffer
    val a = new Array[Int](content.length)
    b.get(a)
    assertEquals(content, a.toSeq)
  }

  test("+") {
    assertEquals((Tensor(1) + 2).item, 3)
  }

  test("grad") {
    val t = torch.ones(Seq(3)) * 2
    t.requiresGrad = true
    val sum = t.sum
    assertEquals(t.grad.dtype, undefined)
    sum.backward()
    assertEquals(t.grad.dtype, float32)
    assert(t.grad.equal(torch.ones(Seq(3))))
  }

  // TODO addcdiv
  // TODO addcmul
  // TODO angle

  testUnaryOp(
    op = asin,
    opName = "asin",
    inputTensor = Tensor(Seq(-0.5962, 1.4985, -0.4396, 1.4525)),
    expectedTensor = Tensor(Seq(-0.6387, Double.NaN, -0.4552, Double.NaN))
  )
  testUnaryOp(
    op = asinh,
    opName = "asinh",
    inputTensor = Tensor(Seq(0.1606, -1.4267, -1.0899, -1.0250)),
    expectedTensor = Tensor(Seq(0.1599, -1.1534, -0.9435, -0.8990))
  )
  testUnaryOp(
    op = atan,
    opName = "atan",
    inputTensor = Tensor(Seq(0.2341, 0.2539, -0.6256, -0.6448)),
    expectedTensor = Tensor(Seq(0.2299, 0.2487, -0.5591, -0.5727))
  )
  testUnaryOp(
    op = atanh,
    opName = "atanh",
    inputTensor = Tensor(Seq(-0.9385, 0.2968, -0.8591, -0.1871)),
    expectedTensor = Tensor(Seq(-1.7253, 0.3060, -1.2899, -0.1893))
  )

  // TODO atan2

  // TODO Test boolean cases for bitwise_not
  // https://pytorch.org/docs/stable/generated/torch.bitwise_not.html
  testUnaryOp(
    op = bitwiseNot,
    opName = "bitwiseNot",
    inputTensor = Tensor(Seq(-1, -2, 3)),
    expectedTensor = Tensor(Seq(0, 1, -4))
  )

  // TODO bitwise_and
  // TODO bitwise_or
  // TODO bitwise_xor
  // TODO bitwise_left_shift
  // TODO bitwise_right_shift

  testUnaryOp(
    op = ceil,
    opName = "ceil",
    inputTensor = Tensor(Seq(-0.6341, -1.4208, -1.0900, 0.5826)),
    expectedTensor = Tensor(Seq(-0.0, -1.0, -1.0, 1.0))
  )

  // TODO clamp

  // TODO Handle Complex Tensors
  // testUnaryOp(
  //   op = conjPhysical,
  //   opName = "conjPhysical",
  //   inputTensor = Tensor(Seq(180.0, -180.0, 360.0, -360.0, 90.0, -90.0)),
  //   expectedTensor = Tensor(Seq(3.1416, -3.1416, 6.2832, -6.2832, 1.5708, -1.5708))
  // )

  // TODO copysign

  testUnaryOp(
    op = cos,
    opName = "cos",
    inputTensor = Tensor(Seq(1.4309, 1.2706, -0.8562, 0.9796)),
    expectedTensor = Tensor(Seq(0.1395, 0.2957, 0.6553, 0.5574))
  )

  testUnaryOp(
    op = cosh,
    opName = "cosh",
    inputTensor = Tensor(Seq(0.1632, 1.1835, -0.6979, -0.7325)),
    expectedTensor = Tensor(Seq(1.0133, 1.7860, 1.2536, 1.2805))
  )

  testUnaryOp(
    op = deg2rad,
    opName = "deg2rad",
    inputTensor = Tensor(Seq(180.0, -180.0, 360.0, -360.0, 90.0, -90.0)),
    expectedTensor = Tensor(Seq(3.1416, -3.1416, 6.2832, -6.2832, 1.5708, -1.5708))
  )

  // TODO div

  testUnaryOp(
    op = digamma,
    opName = "digamma",
    inputTensor = Tensor(Seq(1, 0.5)),
    expectedTensor = Tensor(Seq(-0.5772, -1.9635))
  )

  testUnaryOp(
    op = erf,
    opName = "erf",
    inputTensor = Tensor(Seq(0, -1.0, 10.0)),
    expectedTensor = Tensor(Seq(0.0, -0.8427, 1.0))
  )

  testUnaryOp(
    op = erfc,
    opName = "erfc",
    inputTensor = Tensor(Seq(0, -1.0, 10.0)),
    expectedTensor = Tensor(Seq(1.0, 1.8427, 0.0))
  )

  testUnaryOp(
    op = erfinv,
    opName = "erfinv",
    inputTensor = Tensor(Seq(0.0, 0.5, -1.0)),
    expectedTensor = Tensor(Seq(0.0, 0.4769, Double.NegativeInfinity))
  )

  testUnaryOp(
    op = exp,
    opName = "exp",
    inputTensor = Tensor(Seq(0, 0.6931)),
    expectedTensor = Tensor(Seq(1.0, 2.0))
  )

  testUnaryOp(
    op = exp2,
    opName = "exp2",
    inputTensor = Tensor(Seq(0.0, 1.0, 3.0, 4.0)),
    expectedTensor = Tensor(Seq(1.0, 2.0, 8.0, 16.0))
  )

  testUnaryOp(
    op = expm1,
    opName = "expm1",
    inputTensor = Tensor(Seq(0, 0.6931)),
    expectedTensor = Tensor(Seq(0.0, 1.0))
  )

  // TODO fakeQuantizePerChannelAffine
  // TODO fakeQuantizePerTensorAffine
  // TODO floatPower

  testUnaryOp(
    op = floor,
    opName = "floor",
    inputTensor = Tensor(Seq(-0.8166, 1.5308, -0.2530, -0.2091)),
    expectedTensor = Tensor(Seq(-1.0, 1.0, -1.0, -1.0))
  )

  // TODO floorDivide
  // TODO fmod

  testUnaryOp(
    op = frac,
    opName = "frac",
    inputTensor = Tensor(Seq(1, 2.5, -3.2)),
    expectedTensor = Tensor(Seq(0.0, 0.5, -0.2))
  )

  // TODO Handle Tuple Tensor Output
  // https://pytorch.org/docs/stable/generated/torch.frexp.html
  // testUnaryOp(
  //   op = frexp,
  //   opName = "frexp",
  //   inputTensor = Tensor(Seq(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)),
  //   expectedTensor = Tensor(Seq(0.5724,  0.0, -0.1208))
  // )

  // TODO gradient

  // TODO Handle Complex Tensors
  // testUnaryOp(
  //   op = imag,
  //   opName = "imag",
  //   inputTensor = Tensor(Seq((0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j))),
  //   expectedTensor = Tensor(Seq(0.3553, -0.7896, -0.0633, -0.8119))
  // )

  // TODO ldexp
  // TODO lerp

  testUnaryOp(
    op = lgamma,
    opName = "lgamma",
    inputTensor = Tensor(Seq(0.5, 1.0, 1.5)),
    expectedTensor = Tensor(Seq(0.5724, 0.0, -0.1208))
  )

  testUnaryOp(
    op = log,
    opName = "log",
    inputTensor = Tensor(Seq(4.7767, 4.3234, 1.2156, 0.2411, 4.5739)),
    expectedTensor = Tensor(Seq(1.5637, 1.4640, 0.1952, -1.4226, 1.5204))
  )

  testUnaryOp(
    op = log10,
    opName = "log10",
    inputTensor = Tensor(Seq(0.5224, 0.9354, 0.7257, 0.1301, 0.2251)),
    expectedTensor = Tensor(Seq(-0.2820, -0.0290, -0.1392, -0.8857, -0.6476))
  )

  testUnaryOp(
    op = log1p,
    opName = "log1p",
    inputTensor = Tensor(Seq(-1.0090, -0.9923, 1.0249, -0.5372, 0.2492)),
    expectedTensor = Tensor(Seq(Double.NaN, -4.8653, 0.7055, -0.7705, 0.2225)),
    absolutePrecision = 1e-2
  )

  testUnaryOp(
    op = log2,
    opName = "log2",
    inputTensor = Tensor(Seq(0.8419, 0.8003, 0.9971, 0.5287, 0.0490)),
    expectedTensor = Tensor(Seq(-0.2483, -0.3213, -0.0042, -0.9196, -4.3504)),
    absolutePrecision = 1e-2
  )

  // TODO logaddexp
  // TODO logaddexp2
  // TODO logicalAnd

  // TODO Handle numeric cases for logical_not
  // https://pytorch.org/docs/stable/generated/torch.logical_not.html
  testUnaryOp(
    op = logicalNot,
    opName = "logicalNot",
    inputTensor = Tensor(Seq(true, false)),
    expectedTensor = Tensor(Seq(false, true))
  )

  // TODO logicalOr
  // TODO logicalXor
  // TODO logit
  // TODO hypot

  testUnaryOp(
    op = i0,
    opName = "i0",
    inputTensor = Tensor(Seq(0.0, 1.0, 2.0, 3.0, 4.0)),
    expectedTensor = Tensor(Seq(1.0, 1.2661, 2.2796, 4.8808, 11.3019))
  )

  // TODO igamma
  // TODO igammac
  // TODO mul
  // TODO mvlgamma
  // TODO nanToNum

  testUnaryOp(
    op = neg,
    opName = "neg",
    inputTensor = Tensor(Seq(0.0090, -0.2262, -0.0682, -0.2866, 0.3940)),
    expectedTensor = Tensor(Seq(-0.0090, 0.2262, 0.0682, 0.2866, -0.3940))
  )

  // TODO nextafter
  // TODO polygamma

  testUnaryOp(
    op = positive,
    opName = "positive",
    inputTensor = Tensor(Seq(0.0090, -0.2262, -0.0682, -0.2866, 0.3940)),
    expectedTensor = Tensor(Seq(0.0090, -0.2262, -0.0682, -0.2866, 0.3940))
  )

  // TODO pow
  // TODO quantized_batch_norm
  // TODO quantized_max_pool1d
  // TODO quantized_max_pool2d

  testUnaryOp(
    op = rad2Deg,
    opName = "rad2Deg",
    inputTensor = Tensor(Seq(3.142, -3.142, 6.283, -6.283, 1.570, -1.570)),
    expectedTensor = Tensor(Seq(180.0233, -180.0233, 359.9894, -359.9894, 89.9544, -89.9544))
  )

  // TODO Handle Complex Tensors
  // testUnaryOp(
  //   op = real,
  //   opName = "real",
  //   inputTensor = Tensor(Seq((0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j))),
  //   expectedTensor = Tensor(Seq(0.3100, -0.5445, -1.6492, -0.0638))
  // )

  testUnaryOp(
    op = reciprocal,
    opName = "reciprocal",
    inputTensor = Tensor(Seq(-0.4595, -2.1219, -1.4314, 0.7298)),
    expectedTensor = Tensor(Seq(-2.1763, -0.4713, -0.6986, 1.3702))
  )

  // TODO remainder
  // TODO round

  testUnaryOp(
    op = rsqrt,
    opName = "rsqrt",
    inputTensor = Tensor(Seq(-0.0370, 0.2970, 1.5420, -0.9105)),
    expectedTensor = Tensor(Seq(Double.NaN, 1.8351, 0.8053, Double.NaN)),
    absolutePrecision = 1e-3
  )

  // TODO sigmoid

  testUnaryOp(
    op = sign,
    opName = "sign",
    inputTensor = Tensor(Seq(0.7, -1.2, 0.0, 2.3)),
    expectedTensor = Tensor(Seq(1.0, -1.0, 0.0, 1.0))
  )

  // TODO Fix Complex Tensor creation
  // testUnaryOp(
  //   op = sgn,
  //   opName = "sgn",
  //   inputTensor = Tensor(Seq(Complex(3.0,4.0), Complex(7.0, -24.0), Complex(0.0, 0.0), Complex(1.0, 2.0))),
  //   expectedTensor = Tensor(Seq(false, true, false, true, false))
  // )

  testUnaryOp(
    op = signbit,
    opName = "signbit",
    inputTensor = Tensor(Seq(0.7, -1.2, 0.0, -0.0, 2.3)),
    expectedTensor = Tensor(Seq(false, true, false, true, false))
  )

  testUnaryOp(
    op = sin,
    opName = "sin",
    inputTensor = Tensor(Seq(-0.5461, 0.1347, -2.7266, -0.2746)),
    expectedTensor = Tensor(Seq(-0.5194, 0.1343, -0.4032, -0.2711))
  )

  testUnaryOp(
    op = sinc,
    opName = "sinc",
    inputTensor = Tensor(Seq(0.2252, -0.2948, 1.0267, -1.1566)),
    expectedTensor = Tensor(Seq(0.9186, 0.8631, -0.0259, -0.1300))
  )

  testUnaryOp(
    op = sinh,
    opName = "sinh",
    inputTensor = Tensor(Seq(0.5380, -0.8632, -0.1265, 0.9399)),
    expectedTensor = Tensor(Seq(0.5644, -0.9744, -0.1268, 1.0845))
  )

  testUnaryOp(
    op = sqrt,
    opName = "sqrt",
    inputTensor = Tensor(Seq(-2.0755, 1.0226, 0.0831, 0.4806)),
    expectedTensor = Tensor(Seq(Double.NaN, 1.0112, 0.2883, 0.6933))
  )

  testUnaryOp(
    op = square,
    opName = "square",
    inputTensor = Tensor(Seq(-2.0755, 1.0226, 0.0831, 0.4806)),
    expectedTensor = Tensor(Seq(4.3077, 1.0457, 0.0069, 0.2310))
  )

  test("sub") {
    val a = Tensor(Seq(1, 2))
    val b = Tensor(Seq(0, 1))
    val res = sub(a, b)
    assertEquals(res, Tensor(Seq(1, 1)))

    val resAlpha = sub(a, b, alpha = 2)
    assertEquals(
      resAlpha,
      Tensor(Seq(1, 0))
    )
  }

  testUnaryOp(
    op = tan,
    opName = "tan",
    inputTensor = Tensor(Seq(-1.2027, -1.7687, 0.4412, -1.3856)),
    expectedTensor = Tensor(Seq(-2.5930, 4.9859, 0.4722, -5.3366)),
    absolutePrecision = 1e-2
  )

  testUnaryOp(
    op = tanh,
    opName = "tanh",
    inputTensor = Tensor(Seq(0.8986, -0.7279, 1.1745, 0.2611)),
    expectedTensor = Tensor(Seq(0.7156, -0.6218, 0.8257, 0.2553))
  )

  testUnaryOp(
    op = trunc,
    opName = "trunc",
    inputTensor = Tensor(Seq(3.4742, 0.5466, -0.8008, -0.9079)),
    expectedTensor = Tensor(Seq(3.0, 0.0, -0.0, -0.0))
  )

  test("indexing") {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    // first row
    assertEquals(tensor(0), Tensor(Seq(0, 1, 2, 3)))
    // first column
    assertEquals(tensor(torch.Slice(), 0), Tensor(Seq(0, 4, 8, 12)))
    // last column
    assertEquals(tensor(---, -1), Tensor(Seq(3, 7, 11, 15)))
  }
}
