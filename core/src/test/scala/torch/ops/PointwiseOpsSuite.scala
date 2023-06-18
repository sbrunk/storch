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

import spire.math.Complex

class PointwiseOpsSuite extends TensorCheckSuite {

  testUnaryOp(
    op = abs,
    opName = "abs",
    inputTensor = Tensor(Seq(-1, -2, 3)),
    expectedTensor = Tensor(Seq(1, 2, 3))
  )

  testUnaryOp(
    op = acos,
    opName = "acos",
    inputTensor = Tensor(Seq(0.3348, -0.5889, 0.2005, -0.1584)),
    expectedTensor = Tensor(Seq(1.2294, 2.2004, 1.3690, 1.7298))
  )

  testUnaryOp(
    op = acosh,
    opName = "acosh",
    inputTensor = Tensor(Seq(1.3192, 1.9915, 1.9674, 1.7151)),
    expectedTensor = Tensor(Seq(0.7791, 1.3120, 1.2979, 1.1341))
  )

  testUnaryOp(
    op = acosh,
    opName = "acosh",
    inputTensor = Tensor(Seq(1.3192, 1.9915, 1.9674, 1.7151)),
    expectedTensor = Tensor(Seq(0.7791, 1.3120, 1.2979, 1.1341))
  )

  testUnaryOp(
    op = add(_, other = 20),
    opName = "add",
    inputTensor = Tensor(Seq(0.0202, 1.0985, 1.3506, -0.6056)),
    expectedTensor = Tensor(Seq(20.0202, 21.0985, 21.3506, 19.3944))
  )

  // TODO addcdiv
  // TODO addcmul

  testUnaryOp(
    op = angle,
    opName = "angle",
    inputTensor = Tensor(Seq(Complex(-1.0, 1.0), Complex(-2.0, 2.0), Complex(3.0, -3.0))),
    expectedTensor = Tensor(Seq(2.3562, 2.3562, -0.7854))
  )

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

  testBinaryOp(
    op = atan2,
    opName = "atan2",
    inputTensors = (
      Tensor(Seq(0.9041, 0.0196, -0.3108, -2.4423)),
      Tensor(Seq(1.3104, -1.5804, 0.6674, 0.7710))
    ),
    expectedTensor = Tensor(Seq(0.6039, 3.1292, -0.4358, -1.2650))
  )

  // TODO Test boolean cases for bitwise operations

  testUnaryOp(
    op = bitwiseNot,
    opName = "bitwiseNot",
    inputTensor = Tensor(Seq(-1, -2, 3)),
    expectedTensor = Tensor(Seq(0, 1, -4))
  )

  testBinaryOp(
    op = bitwiseAnd,
    opName = "bitwiseAnd",
    inputTensors = (
      Tensor(Seq(-1, -2, 3)),
      Tensor(Seq(1, 0, 3))
    ),
    expectedTensor = Tensor(Seq(1, 0, 3))
  )

  testBinaryOp(
    op = bitwiseOr,
    opName = "bitwiseOr",
    inputTensors = (
      Tensor(Seq(-1, -2, 3)),
      Tensor(Seq(1, 0, 3))
    ),
    expectedTensor = Tensor(Seq(-1, -2, 3))
  )

  testBinaryOp(
    op = bitwiseXor,
    opName = "bitwiseXor",
    inputTensors = (
      Tensor(Seq(-1, -2, 3)),
      Tensor(Seq(1, 0, 3))
    ),
    expectedTensor = Tensor(Seq(-2, -2, 0))
  )

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = bitwiseLeftShift,
    opName = "bitwiseLeftShift",
    inputTensors = (
      Tensor(Seq(-1, -2, 3)),
      Tensor(Seq(1, 0, 3))
    ),
    expectedTensor = Tensor(Seq(-2, -2, 24))
  )

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = bitwiseRightShift,
    opName = "bitwiseRightShift",
    inputTensors = (
      Tensor(Seq(-2, -7, 31)),
      Tensor(Seq(1, 0, 3))
    ),
    expectedTensor = Tensor(Seq(-1, -7, 3))
  )

  testUnaryOp(
    op = ceil,
    opName = "ceil",
    inputTensor = Tensor(Seq(-0.6341, -1.4208, -1.0900, 0.5826)),
    expectedTensor = Tensor(Seq(-0.0, -1.0, -1.0, 1.0))
  )

  // TODO test min max inputs
  testUnaryOp(
    op = clamp(_, min = Some(-0.5), max = Some(0.5)),
    opName = "clamp",
    inputTensor = Tensor(Seq(-1.7120, 0.1734, -0.0478, -0.0922)),
    expectedTensor = Tensor(Seq(-0.5, 0.1734, -0.0478, -0.0922))
  )

  testUnaryOp(
    op = conjPhysical,
    opName = "conjPhysical",
    inputTensor = Tensor(Seq(Complex(-1.0, 1.0), Complex(-2.0, 2.0), Complex(3.0, -3.0))),
    expectedTensor = Tensor(Seq(Complex(-1.0, -1.0), Complex(-2.0, -2.0), Complex(3.0, 3.0)))
  )

  testBinaryOp(
    op = copysign,
    opName = "copysign",
    inputTensors = (
      Tensor(Seq(0.7079, 0.2778, -1.0249, 0.5719)),
      Tensor(Seq(0.2373, 0.3120, 0.3190, -1.1128))
    ),
    expectedTensor = Tensor(Seq(0.7079, 0.2778, 1.0249, -0.5719))
  )

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

  testBinaryOp(
    op = div,
    opName = "div",
    inputTensors = (
      Tensor(Seq(-0.3711, -1.9353, -0.4605, -0.2917)),
      Tensor(Seq(0.8032, 0.2930, -0.8113, -0.2308))
    ),
    expectedTensor = Tensor(Seq(-0.4620, -6.6051, 0.5676, 1.2639))
  )

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

  testUnaryOp(
    op = fix,
    opName = "fix",
    inputTensor = Tensor(Seq(3.4742, 0.5466, -0.8008, -0.9079)),
    expectedTensor = Tensor(Seq(3.0, 0.0, -0.0, -0.0))
  )

  testBinaryOp(
    op = floatPower,
    opName = "floatPower",
    inputTensors = (
      Tensor(Seq(1, 2, 3, 4)),
      Tensor(Seq(2, -3, 4, -5))
    ),
    expectedTensor = Tensor(Seq(1.0, 0.125, 81.0, 9.7656e-4))
  )

  testUnaryOp(
    op = floor,
    opName = "floor",
    inputTensor = Tensor(Seq(-0.8166, 1.5308, -0.2530, -0.2091)),
    expectedTensor = Tensor(Seq(-1.0, 1.0, -1.0, -1.0))
  )

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = floorDivide,
    opName = "floorDivide",
    inputTensors = (
      Tensor(Seq(4.0, 3.0)),
      Tensor(Seq(2.0, 2.0))
    ),
    expectedTensor = Tensor(Seq(2.0, 1.0))
  )

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = fmod,
    opName = "fmod",
    inputTensors = (
      Tensor(Seq(-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)),
      Tensor(Seq(2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
    ),
    expectedTensor = Tensor(Seq(-1.0, -0.0, -1.0, 1.0, 0.0, 1.0))
  )

  testUnaryOp(
    op = frac,
    opName = "frac",
    inputTensor = Tensor(Seq(1, 2.5, -3.2)),
    expectedTensor = Tensor(Seq(0.0, 0.5, -0.2))
  )

  propertyTestUnaryOp(
    op = frexp,
    opName = "frexp"
  )
  test("frexp.unit-test") {
    val input = arange(0.0, 9.0)
    val expectedMantissa =
      Tensor(Seq(0.0, 0.5, 0.5, 0.75, 0.5, 0.6250, 0.75, 0.8750, 0.5)).to(dtype = float32)
    val expectedExponent = Tensor(Seq(0, 1, 2, 2, 3, 3, 3, 3, 4))
    val (mantissa, exponent) = frexp(input)
    assert(
      allclose(mantissa, expectedMantissa) &&
        allclose(exponent, expectedExponent)
    )
  }

  propertyTestUnaryOp(
    op = gradient(_, 1.0, Seq(0), 1),
    opName = "gradient"
  )
  test("gradient.unit-test") {
    val input = Tensor(Seq(1, 2, 4, 8, 10, 20, 40, 80)).view(-1, 4)
    val results = gradient(input, spacing = 1, dim = Seq(0))
    val expectedTensors = Seq(
      Tensor(Seq(9.0, 18.0, 36.0, 72.0, 9.0, 18.0, 36.0, 72.0)).view(-1, 4).to(dtype = float32),
      Tensor(Seq(1.0, 1.5, 3.0, 4.0, 10.0, 15.0, 30.0, 40.0)).view(-1, 4).to(dtype = float32)
    )

    assert(
      results.zip(expectedTensors).forall { (result, expectedTensor) =>
        allclose(result, expectedTensor)
      }
    )
  }

  testUnaryOp(
    op = imag,
    opName = "imag",
    inputTensor = Tensor(
      Seq(
        Complex(0.31, 0.3553),
        Complex(-0.5445, -0.7896),
        Complex(-1.6492, -0.0633),
        Complex(-0.0638, -0.8119)
      )
    ),
    expectedTensor = Tensor(Seq(0.3553, -0.7896, -0.0633, -0.8119))
  )

  testBinaryOp(
    op = ldexp,
    opName = "ldexp",
    inputTensors = (
      Tensor(Seq(1.0)),
      Tensor(Seq(1, 2, 3, 4))
    ),
    expectedTensor = Tensor(Seq(2.0, 4.0, 8.0, 16.0))
  )

  // TODO Test weight as tensor
  // TODO Lerp must accept the same type so we need to fix generators to work properly
  // testBinaryOp(
  //   op = lerp(_, _, weight = 0.5),
  //   opName = "lerp",
  //   inputTensors = (
  //     Tensor(Seq(1.0, 2.0, 3.0, 4.0)),
  //     Tensor(Seq(10.0, 10.0, 10.0, 10.0))
  //   ),
  //   expectedTensor = Tensor(Seq(5.5, 6.0, 6.5, 7.0))
  // )
  unitTestBinaryOp(
    op = lerp(_, _, weight = 0.5),
    opName = "lerp",
    inputTensors = (
      Tensor(Seq(1.0, 2.0, 3.0, 4.0)),
      Tensor(Seq(10.0, 10.0, 10.0, 10.0))
    ),
    expectedTensor = Tensor(Seq(5.5, 6.0, 6.5, 7.0))
  )

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

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = logaddexp,
    opName = "logaddexp",
    inputTensors = (
      Tensor(Seq(-100.0, -200.0, -300.0)),
      Tensor(Seq(-1.0, -2.0, -3.0))
    ),
    expectedTensor = Tensor(Seq(-1.0, -2.0, -3.0))
  )

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = logaddexp2,
    opName = "logaddexp2",
    inputTensors = (
      Tensor(Seq(-100.0, -200.0, -300.0)),
      Tensor(Seq(-1.0, -2.0, -3.0))
    ),
    expectedTensor = Tensor(Seq(-1.0, -2.0, -3.0))
  )

  // TODO Test int32 tensors
  testBinaryOp(
    op = logicalAnd,
    opName = "logicalAnd",
    inputTensors = (
      Tensor(Seq(true, false, true)),
      Tensor(Seq(true, false, false))
    ),
    expectedTensor = Tensor(Seq(true, false, false))
  )

  // TODO Test int32 tensors
  testUnaryOp(
    op = logicalNot,
    opName = "logicalNot",
    inputTensor = Tensor(Seq(true, false)),
    expectedTensor = Tensor(Seq(false, true))
  )

  // TODO Test int32 tensors
  testBinaryOp(
    op = logicalOr,
    opName = "logicalOr",
    inputTensors = (
      Tensor(Seq(true, false, true)),
      Tensor(Seq(true, false, false))
    ),
    expectedTensor = Tensor(Seq(true, false, true))
  )

  // TODO Test int32 tensors
  testBinaryOp(
    op = logicalXor,
    opName = "logicalXor",
    inputTensors = (
      Tensor(Seq(true, false, true)),
      Tensor(Seq(true, false, false))
    ),
    expectedTensor = Tensor(Seq(false, false, true))
  )

  testUnaryOp(
    op = logit(_, Some(1e-6)),
    opName = "logit",
    inputTensor = Tensor(Seq(0.2796, 0.9331, 0.6486, 0.1523, 0.6516)),
    expectedTensor = Tensor(Seq(-0.9466, 2.6352, 0.6131, -1.7169, 0.6261)),
    absolutePrecision = 1e-3
  )

  // TODO Enable property test once we figure out to compile properly with AtLeastOneFloat
  unitTestBinaryOp(
    op = hypot,
    opName = "hypot",
    inputTensors = (Tensor(Seq(4.0)), Tensor(Seq(3.0, 4.0, 5.0))),
    expectedTensor = Tensor(Seq(5.0, 5.6569, 6.4031))
  )

  testUnaryOp(
    op = i0,
    opName = "i0",
    inputTensor = Tensor(Seq(0.0, 1.0, 2.0, 3.0, 4.0)),
    expectedTensor = Tensor(Seq(1.0, 1.2661, 2.2796, 4.8808, 11.3019))
  )

  // TODO Enable property test once we figure out to compile properly with AtLeastOneFloat
  unitTestBinaryOp(
    op = igamma,
    opName = "igamma",
    inputTensors = (
      Tensor(Seq(4.0)),
      Tensor(Seq(3.0, 4.0, 5.0))
    ),
    expectedTensor = Tensor(Seq(0.3528, 0.5665, 0.7350))
  )

  // TODO Enable property test once we figure out to compile properly with AtLeastOneFloat
  unitTestBinaryOp(
    op = igammac,
    opName = "igammac",
    inputTensors = (
      Tensor(Seq(4.0)),
      Tensor(Seq(3.0, 4.0, 5.0))
    ),
    expectedTensor = Tensor(Seq(0.6472, 0.4335, 0.2650))
  )

  testBinaryOp(
    op = mul,
    opName = "mul",
    inputTensors = (
      Tensor(Seq(1.1207)),
      Tensor(Seq(0.5146, 0.1216, -0.5244, 2.2382))
    ),
    expectedTensor = Tensor(Seq(0.5767, 0.1363, -0.5877, 2.5083))
  )

  testUnaryOp(
    op = mvlgamma(_, p = 2),
    opName = "mvlgamma",
    inputTensor = Tensor(Seq(1.6835, 1.8474, 1.1929)),
    expectedTensor = Tensor(Seq(0.3928, 0.4007, 0.7586))
  )

  // TODO Test nan, posinf, neginf arguments
  // TODO Test float32
  testUnaryOp(
    op = nanToNum(_, nan = None, posinf = None, neginf = None),
    opName = "nanToNum",
    inputTensor = Tensor(Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity, 3.14)),
    expectedTensor = Tensor(Seq(0.0, 1.7976931348623157e308, -1.7976931348623157e308, 3.14))
  )

  testUnaryOp(
    op = neg,
    opName = "neg",
    inputTensor = Tensor(Seq(0.0090, -0.2262, -0.0682, -0.2866, 0.3940)),
    expectedTensor = Tensor(Seq(-0.0090, 0.2262, 0.0682, 0.2866, -0.3940))
  )

  // TODO Enable property test once we figure out to compile properly with AtLeastOneFloat
  // TODO Fix this unit test, as is not really significant due to fp precision
  unitTestBinaryOp(
    op = nextafter,
    opName = "nextafter",
    inputTensors = (
      Tensor(Seq(1.0, 2.0)),
      Tensor(Seq(2.0, 1.0))
    ),
    expectedTensor = Tensor(Seq(1.0, 2.0)),
    absolutePrecision = 1e-8
  )

  // TODO Test multiple values of `n`
  testUnaryOp(
    op = polygamma(1, _),
    opName = "polygamma",
    inputTensor = Tensor(Seq(1.0, 0.5)),
    expectedTensor = Tensor(Seq(1.64493, 4.9348))
  )

  testUnaryOp(
    op = positive,
    opName = "positive",
    inputTensor = Tensor(Seq(0.0090, -0.2262, -0.0682, -0.2866, 0.3940)),
    expectedTensor = Tensor(Seq(0.0090, -0.2262, -0.0682, -0.2866, 0.3940))
  )

  // TODO Test scalar exponent
  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  unitTestBinaryOp(
    op = pow,
    opName = "pow",
    inputTensors = (
      Tensor(Seq(1.0, 2.0, 3.0, 4.0)),
      Tensor(Seq(1.0, 2.0, 3.0, 4.0))
    ),
    expectedTensor = Tensor(Seq(1.0, 4.0, 27.0, 256.0))
  )

  // TODO quantized_batch_norm
  // TODO quantized_max_pool1d
  // TODO quantized_max_pool2d

  testUnaryOp(
    op = rad2Deg,
    opName = "rad2Deg",
    inputTensor = Tensor(Seq(3.142, -3.142, 6.283, -6.283, 1.570, -1.570)),
    expectedTensor = Tensor(Seq(180.0233, -180.0233, 359.9894, -359.9894, 89.9544, -89.9544))
  )

  testUnaryOp(
    op = real,
    opName = "real",
    inputTensor = Tensor(
      Seq(
        Complex(0.31, 0.3553),
        Complex(-0.5445, -0.7896),
        Complex(-1.6492, -0.0633),
        Complex(-0.0638, -0.8119)
      )
    ),
    expectedTensor = Tensor(Seq(0.3100, -0.5445, -1.6492, -0.0638))
  )

  testUnaryOp(
    op = reciprocal,
    opName = "reciprocal",
    inputTensor = Tensor(Seq(-0.4595, -2.1219, -1.4314, 0.7298)),
    expectedTensor = Tensor(Seq(-2.1763, -0.4713, -0.6986, 1.3702))
  )

  // TODO Enable property test once we figure out to consider OnlyOneBool evidence in genDType
  // propertyTestBinaryOp(remainder, "remainder")
  test("remainder.unit-test") {
    val result = remainder(Tensor(Seq(-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)), 2)
    val expected = Tensor(Seq(1.0, 0.0, 1.0, 1.0, 0.0, 1.0))
    assert(allclose(result, expected))

    val result2 = remainder(-1.5, Tensor(Seq(1, 2, 3, 4, 5))).to(dtype = float64)
    val expected2 = Tensor(Seq(0.5, 0.5, 1.5, 2.5, 3.5))
    assert(allclose(result2, expected2))

    val result3 = remainder(Tensor(Seq(1, 2, 3, 4, 5)), Tensor(Seq(1, 2, 3, 4, 5)))
    val expected3 = Tensor(Seq(0, 0, 0, 0, 0))

    assert(allclose(result3, expected3))
  }

  testUnaryOp(
    op = round(_, decimals = 0),
    opName = "round",
    inputTensor = Tensor(Seq(4.7, -2.3, 9.1, -7.7)),
    expectedTensor = Tensor(Seq(5.0, -2.0, 9.0, -8.0))
  )
  test("round.unit-test.decimals") {
    val input = Tensor(Seq(0.1234567))
    val result = round(input, decimals = 3)
    assert(allclose(result, Tensor(Seq(0.123)), atol = 1e-3))
  }

  testUnaryOp(
    op = rsqrt,
    opName = "rsqrt",
    inputTensor = Tensor(Seq(-0.0370, 0.2970, 1.5420, -0.9105)),
    expectedTensor = Tensor(Seq(Double.NaN, 1.8351, 0.8053, Double.NaN)),
    absolutePrecision = 1e-3
  )

  testUnaryOp(
    op = sigmoid,
    opName = "sigmoid",
    inputTensor = Tensor(Seq(0.9213, 1.0887, -0.8858, -1.7683)),
    expectedTensor = Tensor(Seq(0.7153, 0.7481, 0.2920, 0.1458))
  )

  testUnaryOp(
    op = sign,
    opName = "sign",
    inputTensor = Tensor(Seq(0.7, -1.2, 0.0, 2.3)),
    expectedTensor = Tensor(Seq(1.0, -1.0, 0.0, 1.0))
  )

  testUnaryOp(
    op = sgn,
    opName = "sgn",
    inputTensor =
      Tensor(Seq(Complex(3.0, 4.0), Complex(7.0, -24.0), Complex(0.0, 0.0), Complex(1.0, 2.0))),
    expectedTensor = Tensor(
      Seq(Complex(0.6, 0.8), Complex(0.28, -0.96), Complex(0.0, 0.0), Complex(0.4472, 0.8944))
    )
  )

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

  testBinaryOp(
    op = sub,
    opName = "sub",
    inputTensors = (
      Tensor(Seq(1, 2)),
      Tensor(Seq(0, 1))
    ),
    expectedTensor = Tensor(Seq(1, 1))
  )
  test("sub.unit-test.alpha") {
    val a = Tensor(Seq(1, 2))
    val b = Tensor(Seq(0, 1))
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

  testBinaryOp(
    op = trueDivide,
    opName = "trueDivide",
    inputTensors = (
      Tensor(Seq(-0.3711, -1.9353, -0.4605, -0.2917)),
      Tensor(Seq(0.8032, 0.2930, -0.8113, -0.2308))
    ),
    expectedTensor = Tensor(Seq(-0.4620, -6.6051, 0.5676, 1.2639))
  )

  testUnaryOp(
    op = trunc,
    opName = "trunc",
    inputTensor = Tensor(Seq(3.4742, 0.5466, -0.8008, -0.9079)),
    expectedTensor = Tensor(Seq(3.0, 0.0, -0.0, -0.0))
  )

  testBinaryOp(
    op = xlogy,
    opName = "xlogy",
    inputTensors = (
      Tensor(Seq(0, 0, 0, 0, 0)),
      Tensor(Seq(-1.0, 0.0, 1.0, Double.PositiveInfinity, Double.NaN))
    ),
    expectedTensor = Tensor(Seq(0.0, 0.0, 0.0, 0.0, Double.NaN))
  )

}
