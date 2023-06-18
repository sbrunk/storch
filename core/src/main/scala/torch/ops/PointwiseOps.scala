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

import internal.NativeConverters.*
import org.bytedeco.pytorch.global.torch as torchNative

/** Pointwise Ops
  *
  * https://pytorch.org/docs/stable/torch.html#pointwise-ops
  */

/** Computes the absolute value of each element in `input`. */
def abs[D <: NumericNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.abs(input.native))

/** Computes the inverse cosine of each element in `input`. */
def acos[D <: DType](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.acos(input.native))

/** Returns a new tensor with the inverse hyperbolic cosine of the elements of `input` . */
def acosh[D <: DType](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.acosh(input.native))

/** Adds `other` to `input`. */
def add[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.add(input.native, other.native))

/** Adds `other` to `input`. */
def add[D <: DType, S <: ScalaType](
    input: Tensor[D],
    other: S
): Tensor[Promoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.add(input.native, toScalar(other)))

/** Performs the element-wise division of tensor1 by tensor2, multiplies the result by the scalar
  * value and adds it to input.
  */
def addcdiv[D <: DType, D2 <: DType, D3 <: DType](
    input: Tensor[D],
    tensor1: Tensor[D2],
    tensor2: Tensor[D3],
    value: ScalaType
): Tensor[Promoted[D, Promoted[D2, D3]]] =
  Tensor(torchNative.addcdiv(input.native, tensor1.native, tensor2.native, toScalar(value)))

/** Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by the
  * scalar value and adds it to input.
  */
def addcmul[D <: DType, D2 <: DType, D3 <: DType](
    input: Tensor[D],
    tensor1: Tensor[D2],
    tensor2: Tensor[D3],
    value: ScalaType
): Tensor[Promoted[D, Promoted[D2, D3]]] =
  Tensor(torchNative.addcmul(input.native, tensor1.native, tensor2.native, toScalar(value)))

/** Computes the element-wise angle (in radians) of the given `input` tensor. */
def angle[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[ComplexToReal[D]]] =
  Tensor(torchNative.angle(input.native))

/** Returns a new tensor with the arcsine of the elements of `input`. */
def asin[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.asin(input.native))

/** Returns a new tensor with the inverse hyperbolic sine of the elements of `input`. */
def asinh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.asinh(input.native))

/** Returns a new tensor with the arctangent of the elements of `input`. */
def atan[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.atan(input.native))

/** Returns a new tensor with the inverse hyperbolic tangent of the elements of `input`. */
def atanh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.atanh(input.native))

/** Element-wise arctangent of (input / other) with consideration of the quadrant. Returns a new
  * tensor with the signed angles in radians between vector (other, input) and vector (1, 0). (Note
  * that other, the second parameter, is the x-coordinate, while input, the first parameter, is the
  * y-coordinate.)
  */
def atan2[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.atan2(input.native, other.native))

/** Computes the bitwise NOT of the given `input` tensor. The `input` tensor must be of integral or
  * Boolean types. For bool tensors, it computes the logical NOT.
  */
def bitwiseNot[D <: BitwiseNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_not(input.native))

/** Computes the bitwise AND of `input` and `other`. For bool tensors, it computes the logical AND.
  */
def bitwiseAnd[D <: BitwiseNN, D2 <: BitwiseNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.bitwise_and(input.native, other.native))

/** Computes the bitwise OR of `input` and `other`. For bool tensors, it computes the logical OR.
  */
def bitwiseOr[D <: BitwiseNN, D2 <: BitwiseNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.bitwise_or(input.native, other.native))

/** Computes the bitwise XOR of `input` and `other`. For bool tensors, it computes the logical XOR.
  */
def bitwiseXor[D <: BitwiseNN, D2 <: BitwiseNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.bitwise_xor(input.native, other.native))

/** Computes the left arithmetic shift of `input` by `other` bits. */

def bitwiseLeftShift[D <: BitwiseNN, D2 <: BitwiseNN](
    input: Tensor[D],
    other: Tensor[D2]
)(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.bitwise_left_shift(input.native, other.native))

/** Computes the right arithmetic s\hift of `input` by `other` bits. */
def bitwiseRightShift[D <: BitwiseNN, D2 <: BitwiseNN](
    input: Tensor[D],
    other: Tensor[D2]
)(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.bitwise_right_shift(input.native, other.native))

/** Returns a new tensor with the ceil of the elements of `input`, the smallest integer greater than
  * or equal to each element.
  */
def ceil[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.ceil(input.native))

/** Clamps all elements in `input` into the range [ min, max ]. Letting min_value and max_value be
  * min and max, respectively, this returns: `min(max(input, min_value), max_value)` If min is None,
  * there is no lower bound. Or, if max is None there is no upper bound.
  */
// TODO Support Tensor for min and max
def clamp[D <: RealNN](
    input: Tensor[D],
    min: Option[Real],
    max: Option[Real]
): Tensor[D] =
  Tensor(torchNative.clamp(input.native, toOptional(min), toOptional(max)))

/** Computes the element-wise conjugate of the given `input` tensor. If input has a non-complex
  * dtype, this function just returns input.
  */
def conjPhysical[D <: DType](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.conj_physical(input.native))

/** Create a new floating-point tensor with the magnitude of input and the sign of other,
  * elementwise.
  */
def copysign[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: TensorOrReal[D2]
): Tensor[FloatPromoted[D]] =
  Tensor(
    other match
      case other: Tensor[D2] =>
        torchNative.copysign(input.native, other.native)
      case other: Real =>
        torchNative.copysign(input.native, toScalar(other))
  )

/** Returns a new tensor with the cosine of the elements of `input`. */
def cos[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.cos(input.native))

/** Returns a new tensor with the hyperbolic cosine of the elements of `input`. */
def cosh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.cosh(input.native))

/** Returns a new tensor with each of the elements of `input` converted from angles in degrees to
  * radians.
  */
def deg2rad[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.deg2rad(input.native))

/** Divides each element of the input `input` by the corresponding element of `other`. */
// TODO handle roundingMode
def div[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.div(input.native, other.native))

def div[D <: DType, S <: ScalaType](
    input: Tensor[D],
    other: S
): Tensor[FloatPromoted[Promoted[D, ScalaToDType[S]]]] =
  Tensor(torchNative.div(input.native, toScalar(other)))

export torch.special.digamma
export torch.special.erf
export torch.special.erfc
export torch.special.erfinv

/** Returns a new tensor with the exponential of the elements of the `input` tensor `input`. */
def exp[D <: DType](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.exp(input.native))

export torch.special.exp2
export torch.special.expm1

/** Returns a new tensor with the data in `input` fake quantized per channel using `scale`,
  * `zero_point`, `quant_min` and `quant_max`, across the channel specified by `axis`.
  */
def fakeQuantizePerChannelAffine(
    input: Tensor[Float32],
    scale: Tensor[Float32],
    zeroPoint: Tensor[Int32 | Float16 | Float32],
    axis: Long,
    quantMin: Long,
    quantMax: Long
): Tensor[Float32] =
  Tensor(
    torchNative.fake_quantize_per_channel_affine(
      input.native,
      scale.native,
      zeroPoint.native,
      axis,
      quantMin,
      quantMax
    )
  )

/** Returns a new tensor with the data in `input` fake quantized using `scale`, `zero_point`,
  * `quant_min` and `quant_max`.
  */
def fakeQuantizePerTensorAffine(
    input: Tensor[Float32],
    scale: Tensor[Float32],
    zeroPoint: Tensor[Int32],
    quantMin: Long,
    quantMax: Long
): Tensor[Float32] =
  Tensor(
    torchNative.fake_quantize_per_tensor_affine(
      input.native,
      scale.native,
      zeroPoint.native,
      quantMin,
      quantMax
    )
  )

def fakeQuantizePerTensorAffine(
    input: Tensor[Float32],
    scale: Double,
    zeroPoint: Long,
    quantMin: Long,
    quantMax: Long
): Tensor[Float32] =
  Tensor(
    torchNative.fake_quantize_per_tensor_affine(input.native, scale, zeroPoint, quantMin, quantMax)
  )

/** Returns a new tensor with the truncated integer values of the elements of `input`. Alias for
  * torch.trunc
  */
def fix[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.fix(input.native))

/** Raises `input` to the power of `exponent`, elementwise, in double precision. If neither input is
  * complex returns a `torch.float64` tensor, and if one or more inputs is complex returns a
  * `torch.complex128` tensor.
  */
def floatPower[D <: DType, D2 <: DType](
    input: Tensor[D],
    exponent: Tensor[D2]
): Tensor[ComplexPromoted[D, D2]] =
  Tensor(torchNative.float_power(input.native, exponent.native))

def floatPower[D <: DType, S <: ScalaType](
    input: S,
    exponent: Tensor[D]
): Tensor[ComplexPromoted[ScalaToDType[S], D]] =
  Tensor(torchNative.float_power(toScalar(input), exponent.native))

def floatPower[D <: DType, S <: ScalaType](
    input: Tensor[D],
    exponent: ScalaType
): Tensor[ComplexPromoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.float_power(input.native, toScalar(exponent)))

/** Returns a new tensor with the floor of the elements of `input`, the largest integer less than or
  * equal to each element.
  */
def floor[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.floor(input.native))

/** Computes `input` divided by `other`, elementwise, and floors the result. */
def floorDivide[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: Tensor[D2]
)(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.floor_divide(input.native, other.native))

def floorDivide[D <: RealNN, R <: Real](
    input: Tensor[D],
    other: R
)(using OnlyOneBool[D, ScalaToDType[R]]): Tensor[Promoted[D, ScalaToDType[R]]] =
  Tensor(torchNative.floor_divide(input.native, toScalar(other)))

/** Applies C++’s `std::fmod` entrywise. The result has the same sign as the dividend `input` and
  * its absolute value is less than that of `other`.
  */
// NOTE: When the divisor is zero, returns NaN for floating point dtypes on both CPU and GPU; raises RuntimeError for integer division by zero on CPU; Integer division by zero on GPU may return any value.
def fmod[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: Tensor[D2]
)(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.fmod(input.native, other.native))

def fmod[D <: RealNN, S <: ScalaType](
    input: Tensor[D],
    other: S
)(using OnlyOneBool[D, ScalaToDType[S]]): Tensor[Promoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.fmod(input.native, toScalar(other)))

/** Computes the fractional portion of each element in `input`. */
def frac[D <: FloatNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.frac(input.native))

/** Decomposes `input` into `mantissa` and `exponent` tensors such that `input = mantissa * (2 **
  * exponent)` The range of mantissa is the open interval (-1, 1).
  */
def frexp[D <: FloatNN](input: Tensor[D]): (Tensor[FloatPromoted[D]], Tensor[Int32]) =
  val nativeTuple = torchNative.frexp(input.native)
  (Tensor(nativeTuple.get0), new Int32Tensor(nativeTuple.get1))

/** Estimates the gradient of a function g:Rn → R in one or more dimensions using the second-order
  * accurate central differences method.
  */
def gradient[D <: Int8 | Int16 | Int32 | Int64 | FloatNN | ComplexNN](
    input: Tensor[D],
    spacing: Float,
    dim: Seq[Int],
    edgeOrder: Int = 1
): Array[Tensor[D]] =
  torchNative
    .gradient(input.native, toScalar(spacing), dim.toArray.map(_.toLong), edgeOrder)
    .get
    .map(Tensor.apply[D])

/** Returns a new tensor containing imaginary values of the `input` tensor. The returned tensor and
  * `input` share the same underlying storage.
  */
def imag[D <: ComplexNN](input: Tensor[D]): Tensor[ComplexToReal[D]] =
  Tensor(torchNative.imag(input.native))

/** Multiplies `input` by 2 ** `other`. */
def ldexp[D <: DType](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.ldexp(input.native, other.native))

/** Does a linear interpolation of two tensors `start` (given by `input`) and `end` (given by
  * `other`) based on a scalar or tensor weight and returns the resulting out tensor. out = start +
  * weight × (end − start)
  */
def lerp[D <: DType](
    input: Tensor[D],
    other: Tensor[D],
    weight: Tensor[D] | Float | Double
): Tensor[D] =
  Tensor(
    weight match
      case weight: Tensor[D] => torchNative.lerp(input.native, other.native, weight.native)
      case weight: Float     => torchNative.lerp(input.native, other.native, toScalar(weight))
      case weight: Double    => torchNative.lerp(input.native, other.native, toScalar(weight))
  )

/** Computes the natural logarithm of the absolute value of the gamma function on `input`. */
def lgamma[D <: RealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.lgamma(input.native))

/** Returns a new tensor with the natural logarithm of the elements of `input`. */
def log[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log(input.native))

/** Returns a new tensor with the logarithm to the base 10 of the elements of `input`. */
def log10[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log10(input.native))

/** Returns a new tensor with the natural logarithm of (1 + input). */
def log1p[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log1p(input.native))

/** Returns a new tensor with the logarithm to the base 2 of the elements of `input`. */
def log2[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log2(input.native))

/** Logarithm of the sum of exponentiations of the inputs. Calculates pointwise log `log(e**x +
  * e**y)`. This function is useful in statistics where the calculated probabilities of events may
  * be so small as to exceed the range of normal floating point numbers. In such cases the logarithm
  * of the calculated probability is stored. This function allows adding probabilities stored in
  * such a fashion. This op should be disambiguated with `torch.logsumexp()` which performs a
  * reduction on a single tensor.
  */
def logaddexp[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.logaddexp(input.native, other.native))

/** Logarithm of the sum of exponentiations of the inputs in base-2. Calculates pointwise `log2(2**x
  * + 2**y)`. See torch.logaddexp() for more details.
  */
def logaddexp2[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.logaddexp2(input.native, other.native))

/** Computes the element-wise logical AND of the given `input` tensors. Zeros are treated as False
  * and nonzeros are treated as True.
  */
def logicalAnd[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
  Tensor(torchNative.logical_and(input.native, other.native))

/** Computes the element-wise logical NOT of the given `input` tensor. If the `input` tensor is not
  * a bool tensor, zeros are treated as False and non-zeros are treated as True.
  *
  * TODO If not specified, the output tensor will have the bool dtype.
  */
def logicalNot[D <: DType](input: Tensor[D]): Tensor[Bool] =
  Tensor(torchNative.logical_not(input.native))

/** Computes the element-wise logical OR of the given `input` tensors. Zeros are treated as False
  * and nonzeros are treated as True.
  */
def logicalOr[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
  Tensor(torchNative.logical_or(input.native, other.native))

/** Computes the element-wise logical XOR of the given `input` tensors. Zeros are treated as False
  * and nonzeros are treated as True.
  */
def logicalXor[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
  Tensor(torchNative.logical_xor(input.native, other.native))

export torch.special.logit

/** Given the legs of a right triangle, return its hypotenuse. */
// TODO Change `D2 <: RealNN` once we fix property testing compilation
def hypot[D <: RealNN, D2 <: FloatNN](
    input: Tensor[D],
    other: Tensor[D2]
)(using AtLeastOneFloat[D, D2]): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.hypot(input.native, other.native))

export torch.special.i0
export torch.special.igamma
export torch.special.igammac

/** Multiplies input by other. */
def mul[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.mul(input.native, other.native))

export torch.special.mvlgamma

/** Replaces NaN, positive infinity, and negative infinity values in `input` with the values
  * specified by nan, posinf, and neginf, respectively. By default, NaNs are replaced with zero,
  * positive infinity is replaced with the greatest finite value representable by input’s dtype, and
  * negative infinity is replaced with the least finite value representable by input’s dtype.
  */
def nanToNum[D <: RealNN](
    input: Tensor[D],
    nan: Option[Double] = None,
    posinf: Option[Double] = None,
    neginf: Option[Double] = None
): Tensor[D] =
  Tensor(
    torchNative.nan_to_num(input.native, toOptional(nan), toOptional(posinf), toOptional(neginf))
  )

/** Returns a new tensor with the negative of the elements of `input`. */
def neg[D <: NumericNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.neg(input.native))

/** Return the next floating-point value after `input` towards `other`, elementwise. */
// TODO Change `D2 <: RealNN` once we fix property testing compilation
def nextafter[D <: RealNN, D2 <: FloatNN](
    input: Tensor[D],
    other: Tensor[D2]
)(using AtLeastOneFloat[D, D2]): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.nextafter(input.native, other.native))

export torch.special.polygamma

/** Returns input. Normally throws a runtime error if input is a bool tensor in pytorch. */
def positive[D <: NumericNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.positive(input.native))

/** Takes the power of each element in `input` with exponent and returns a tensor with the result.
  * `exponent` can be either a single float number or a Tensor with the same number of elements as
  * input.
  */
def pow[D <: DType, D2 <: DType](
    input: Tensor[D],
    exponent: Tensor[D2]
)(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.pow(input.native, exponent.native))

def pow[D <: DType, S <: ScalaType](
    input: Tensor[D],
    exponent: S
)(using OnlyOneBool[D, ScalaToDType[S]]): Tensor[Promoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.pow(input.native, toScalar(exponent)))

def pow[S <: ScalaType, D <: DType](
    input: S,
    exponent: Tensor[D]
)(using OnlyOneBool[ScalaToDType[S], D]): Tensor[Promoted[ScalaToDType[S], D]] =
  Tensor(torchNative.pow(toScalar(input), exponent.native))

// TODO Implement creation of QInts
// TODO quantized_batch_norm
// TODO quantized_max_pool1d
// TODO quantized_max_pool2d

/** Returns a new tensor with each of the elements of `input` converted from angles in radians to
  * degrees.
  */
def rad2Deg[D <: RealNN | Bool](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.rad2deg(input.native))

/** Returns a new tensor containing real values of the self tensor. The returned tensor and self
  * share the same underlying storage.
  */
def real[D <: DType](input: Tensor[D]): Tensor[ComplexToReal[D]] =
  Tensor(torchNative.real(input.native))

/** Returns a new tensor with the reciprocal of the elements of `input` */
def reciprocal[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.reciprocal(input.native))

/** Computes Python’s modulus operation entrywise. The result has the same sign as the divisor
  * `other` and its absolute value is less than that of `other`.
  */
def remainder[D <: RealNN, D2 <: RealNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.remainder(input.native, other.native))

def remainder[D <: DType, R <: Real](
    input: Tensor[D],
    other: R
): Tensor[Promoted[D, ScalaToDType[R]]] =
  Tensor(torchNative.remainder(input.native, toScalar(other)))

def remainder[D <: DType, R <: Real](
    input: R,
    other: Tensor[D]
): Tensor[Promoted[ScalaToDType[R], D]] =
  Tensor(torchNative.remainder(toScalar(input), other.native))

/** Rounds elements of `input` to the nearest integer. If decimals is negative, it specifies the
  * number of positions to the left of the decimal point.
  */
def round[D <: FloatNN](input: Tensor[D], decimals: Long = 0): Tensor[D] =
  Tensor(torchNative.round(input.native, decimals))

/** Returns a new tensor with the reciprocal of the square-root of each of the elements of `input`.
  */
def rsqrt[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.rsqrt(input.native))

export torch.special.sigmoid

/** Returns a new tensor with the signs of the elements of `input`. */
def sign[D <: RealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.sign(input.native))

/** This function is an extension of `torch.sign()` to complex tensors. It computes a new tensor
  * whose elements have the same angles as the corresponding elements of `input` and absolute values
  * (i.e. magnitudes) of one for complex tensors and is equivalent to torch.sign() for non-complex
  * tensors.
  */
def sgn[D <: DType](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.sgn(input.native))

/** Tests if each element of `input`` has its sign bit set or not. */
def signbit[D <: RealNN](input: Tensor[D]): Tensor[Bool] =
  Tensor(torchNative.signbit(input.native))

/** Returns a new tensor with the sine of the elements of `input`. */
def sin[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.sin(input.native))

export torch.special.sinc

/** Returns a new tensor with the hyperbolic sine of the elements of `input`. */
def sinh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.sinh(input.native))

export torch.nn.functional.softmax

/** Returns a new tensor with the square-root of the elements of `input`. */
def sqrt[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.sqrt(input.native))

/** Returns a new tensor with the square of the elements of `input`. */
def square[D <: DType](input: Tensor[D]): Tensor[NumericPromoted[D]] =
  Tensor(torchNative.square(input.native))

/** Subtracts `other`, scaled by `alpha`, from `input`. */
def sub[D <: NumericNN, D2 <: NumericNN](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.sub(input.native, other.native))

def sub[D <: NumericNN, D2 <: NumericNN](
    input: Tensor[D],
    other: Tensor[D2],
    alpha: ScalaType
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.sub(input.native, other.native, toScalar(alpha)))

def sub[D <: NumericNN, D2 <: NumericNN](
    input: Tensor[D],
    other: Numeric,
    alpha: ScalaType
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.sub(input.native, toScalar(other), toScalar(alpha)))

/** Returns a new tensor with the tangent of the elements of `input`. */
def tan[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.tan(input.native))

/** Returns a new tensor with the hyperbolic tangent of the elements of `input`. */
def tanh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.tanh(input.native))

/** Alias for `torch.div()` with `rounding_mode=None` */
def trueDivide[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.true_divide(input.native, other.native))

def trueDivide[D <: DType, S <: ScalaType](
    input: Tensor[D],
    other: S
): Tensor[FloatPromoted[Promoted[D, ScalaToDType[S]]]] =
  Tensor(torchNative.true_divide(input.native, toScalar(other)))

/** Returns a new tensor with the truncated integer values of the elements of `input`. */
def trunc[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.trunc(input.native))

export torch.special.xlogy
