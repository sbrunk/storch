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
package ops

import internal.NativeConverters.*
import org.bytedeco.pytorch.global.torch as torchNative
import scala.annotation.implicitNotFound

/** Pointwise Ops
  *
  * https://pytorch.org/docs/stable/torch.html#pointwise-ops
  */
private[torch] trait PointwiseOps {

  /** Computes the absolute value of each element in `input`.
    *
    * @group pointwise_ops
    */
  def abs[D <: NumericNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.abs(input.native))

  /** Computes the inverse cosine of each element in `input`.
    *
    * @group pointwise_ops
    */
  def acos[D <: DType](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.acos(input.native))

  /** Returns a new tensor with the inverse hyperbolic cosine of the elements of `input` .
    *
    * @group pointwise_ops
    */
  def acosh[D <: DType](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.acosh(input.native))

  /** Adds `other` to `input`.
    *
    * @group pointwise_ops
    */
  def add[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.add(input.native, other.native))

  /** Adds `other` to `input`.
    *
    * @group pointwise_ops
    */
  def add[D <: DType, S <: ScalaType](
      input: Tensor[D],
      other: S
  ): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(torchNative.add(input.native, toScalar(other)))

  /** Performs the element-wise division of tensor1 by tensor2, multiplies the result by the scalar
    * value and adds it to input.
    *
    * @group pointwise_ops
    */
  def addcdiv[D <: DType, D2 <: DType, D3 <: DType](
      input: Tensor[D],
      tensor1: Tensor[D2],
      tensor2: Tensor[D3],
      value: ScalaType
  ): Tensor[Promoted[D, Promoted[D2, D3]]] =
    fromNative(torchNative.addcdiv(input.native, tensor1.native, tensor2.native, toScalar(value)))

  /** Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by the
    * scalar value and adds it to input.
    *
    * @group pointwise_ops
    */
  def addcmul[D <: DType, D2 <: DType, D3 <: DType](
      input: Tensor[D],
      tensor1: Tensor[D2],
      tensor2: Tensor[D3],
      value: ScalaType
  ): Tensor[Promoted[D, Promoted[D2, D3]]] =
    fromNative(torchNative.addcmul(input.native, tensor1.native, tensor2.native, toScalar(value)))

  /** Computes the element-wise angle (in radians) of the given `input` tensor.
    *
    * @group pointwise_ops
    */
  def angle[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[ComplexToReal[D]]] =
    fromNative(torchNative.angle(input.native))

  /** Returns a new tensor with the arcsine of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def asin[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.asin(input.native))

  /** Returns a new tensor with the inverse hyperbolic sine of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def asinh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.asinh(input.native))

  /** Returns a new tensor with the arctangent of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def atan[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.atan(input.native))

  /** Returns a new tensor with the inverse hyperbolic tangent of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def atanh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.atanh(input.native))

  /** Element-wise arctangent of (input / other) with consideration of the quadrant. Returns a new
    * tensor with the signed angles in radians between vector (other, input) and vector (1, 0).
    * (Note that other, the second parameter, is the x-coordinate, while input, the first parameter,
    * is the y-coordinate.)
    *
    * @group pointwise_ops
    */
  def atan2[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.atan2(input.native, other.native))

  /** Computes the bitwise NOT of the given `input` tensor. The `input` tensor must be of integral
    * or Boolean types. For bool tensors, it computes the logical NOT.
    *
    * @group pointwise_ops
    */
  def bitwiseNot[D <: BitwiseNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.bitwise_not(input.native))

  /** Computes the bitwise AND of `input` and `other`. For bool tensors, it computes the logical
    * AND.
    *
    * @group pointwise_ops
    */
  def bitwiseAnd[D <: BitwiseNN, D2 <: BitwiseNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.bitwise_and(input.native, other.native))

  /** Computes the bitwise OR of `input` and `other`. For bool tensors, it computes the logical OR.
    *
    * @group pointwise_ops
    */
  def bitwiseOr[D <: BitwiseNN, D2 <: BitwiseNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.bitwise_or(input.native, other.native))

  /** Computes the bitwise XOR of `input` and `other`. For bool tensors, it computes the logical
    * XOR.
    *
    * @group pointwise_ops
    */
  def bitwiseXor[D <: BitwiseNN, D2 <: BitwiseNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.bitwise_xor(input.native, other.native))

  /** Computes the left arithmetic shift of `input` by `other` bits.
    *
    * @group pointwise_ops
    */

  def bitwiseLeftShift[D <: BitwiseNN, D2 <: BitwiseNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.bitwise_left_shift(input.native, other.native))

  /** Computes the right arithmetic s\hift of `input` by `other` bits.
    *
    * @group pointwise_ops
    */
  def bitwiseRightShift[D <: BitwiseNN, D2 <: BitwiseNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.bitwise_right_shift(input.native, other.native))

  /** Returns a new tensor with the ceil of the elements of `input`, the smallest integer greater
    * than or equal to each element.
    *
    * @group pointwise_ops
    */
  def ceil[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.ceil(input.native))

  /** Clamps all elements in `input` into the range [ min, max ]. Letting min_value and max_value be
    * min and max, respectively, this returns: `min(max(input, min_value), max_value)` If min is
    * None, there is no lower bound. Or, if max is None there is no upper bound.
    *
    * @group pointwise_ops
    */
// TODO Support Tensor for min and max
  def clamp[D <: RealNN](
      input: Tensor[D],
      min: Option[Real],
      max: Option[Real]
  ): Tensor[D] =
    fromNative(torchNative.clamp(input.native, toOptional(min), toOptional(max)))

  /** Computes the element-wise conjugate of the given `input` tensor. If input has a non-complex
    * dtype, this function just returns input.
    *
    * @group pointwise_ops
    */
  def conjPhysical[D <: DType](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.conj_physical(input.native))

  /** Create a new floating-point tensor with the magnitude of input and the sign of other,
    * elementwise.
    *
    * @group pointwise_ops
    */
  def copysign[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: TensorOrReal[D2]
  ): Tensor[FloatPromoted[D]] =
    fromNative(
      other match
        case other: Tensor[D2] =>
          torchNative.copysign(input.native, other.native)
        case other: Real =>
          torchNative.copysign(input.native, toScalar(other))
    )

  /** Returns a new tensor with the cosine of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def cos[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.cos(input.native))

  /** Returns a new tensor with the hyperbolic cosine of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def cosh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.cosh(input.native))

  /** Returns a new tensor with each of the elements of `input` converted from angles in degrees to
    * radians.
    *
    * @group pointwise_ops
    */
  def deg2rad[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.deg2rad(input.native))

  /** Divides each element of the input `input` by the corresponding element of `other`.
    *
    * @group pointwise_ops
    */
// TODO handle roundingMode
  def div[D <: DType, D2 <: DType](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.div(input.native, other.native))

  def div[D <: DType, S <: ScalaType](
      input: Tensor[D],
      other: S
  ): Tensor[FloatPromoted[Promoted[D, ScalaToDType[S]]]] =
    fromNative(torchNative.div(input.native, toScalar(other)))

  export torch.special.digamma
  export torch.special.erf
  export torch.special.erfc
  export torch.special.erfinv

  /** Returns a new tensor with the exponential of the elements of the `input` tensor `input`.
    *
    * @group pointwise_ops
    */
  def exp[D <: DType](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.exp(input.native))

  export torch.special.exp2
  export torch.special.expm1

  /** Returns a new tensor with the data in `input` fake quantized per channel using `scale`,
    * `zero_point`, `quant_min` and `quant_max`, across the channel specified by `axis`.
    *
    * @group pointwise_ops
    */
  def fakeQuantizePerChannelAffine(
      input: Tensor[Float32],
      scale: Tensor[Float32],
      zeroPoint: Tensor[Int32 | Float16 | Float32],
      axis: Long,
      quantMin: Long,
      quantMax: Long
  ): Tensor[Float32] =
    fromNative(
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
    *
    * @group pointwise_ops
    */
  def fakeQuantizePerTensorAffine(
      input: Tensor[Float32],
      scale: Tensor[Float32],
      zeroPoint: Tensor[Int32],
      quantMin: Long,
      quantMax: Long
  ): Tensor[Float32] =
    fromNative(
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
    fromNative(
      torchNative.fake_quantize_per_tensor_affine(
        input.native,
        scale,
        zeroPoint,
        quantMin,
        quantMax
      )
    )

  /** Returns a new tensor with the truncated integer values of the elements of `input`. Alias for
    * torch.trunc
    *
    * @group pointwise_ops
    */
  def fix[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.fix(input.native))

  /** Raises `input` to the power of `exponent`, elementwise, in double precision. If neither input
    * is complex returns a `torch.float64` tensor, and if one or more inputs is complex returns a
    * `torch.complex128` tensor.
    *
    * @group pointwise_ops
    */
  def floatPower[D <: DType, D2 <: DType](
      input: Tensor[D],
      exponent: Tensor[D2]
  ): Tensor[ComplexPromoted[D, D2]] =
    fromNative(torchNative.float_power(input.native, exponent.native))

  def floatPower[D <: DType, S <: ScalaType](
      input: S,
      exponent: Tensor[D]
  ): Tensor[ComplexPromoted[ScalaToDType[S], D]] =
    fromNative(torchNative.float_power(toScalar(input), exponent.native))

  def floatPower[D <: DType, S <: ScalaType](
      input: Tensor[D],
      exponent: ScalaType
  ): Tensor[ComplexPromoted[D, ScalaToDType[S]]] =
    fromNative(torchNative.float_power(input.native, toScalar(exponent)))

  /** Returns a new tensor with the floor of the elements of `input`, the largest integer less than
    * or equal to each element.
    *
    * @group pointwise_ops
    */
  def floor[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.floor(input.native))

  /** Computes `input` divided by `other`, elementwise, and floors the result.
    *
    * @group pointwise_ops
    */
  def floorDivide[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.floor_divide(input.native, other.native))

  def floorDivide[D <: RealNN, R <: Real](
      input: Tensor[D],
      other: R
  )(using OnlyOneBool[D, ScalaToDType[R]]): Tensor[Promoted[D, ScalaToDType[R]]] =
    fromNative(torchNative.floor_divide(input.native, toScalar(other)))

  /** Applies C++’s `std::fmod` entrywise. The result has the same sign as the dividend `input` and
    * its absolute value is less than that of `other`.
    *
    * @group pointwise_ops
    */
// NOTE: When the divisor is zero, returns NaN for floating point dtypes on both CPU and GPU; raises RuntimeError for integer division by zero on CPU; Integer division by zero on GPU may return any value.
  def fmod[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using OnlyOneBool[D, D2]): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.fmod(input.native, other.native))

  def fmod[D <: RealNN, S <: ScalaType](
      input: Tensor[D],
      other: S
  )(using OnlyOneBool[D, ScalaToDType[S]]): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(torchNative.fmod(input.native, toScalar(other)))

  /** Computes the fractional portion of each element in `input`.
    *
    * @group pointwise_ops
    */
  def frac[D <: FloatNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.frac(input.native))

  /** Decomposes `input` into `mantissa` and `exponent` tensors such that `input = mantissa * (2 **
    * exponent)` The range of mantissa is the open interval (-1, 1).
    *
    * @group pointwise_ops
    */
  def frexp[D <: FloatNN](input: Tensor[D]): (Tensor[FloatPromoted[D]], Tensor[Int32]) =
    val nativeTuple = torchNative.frexp(input.native)
    (fromNative(nativeTuple.get0), new Int32Tensor(nativeTuple.get1))

  /** Estimates the gradient of a function g:Rn → R in one or more dimensions using the second-order
    * accurate central differences method.
    *
    * @group pointwise_ops
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
      .map(fromNative[D])

  /** Returns a new tensor containing imaginary values of the `input` tensor. The returned tensor
    * and `input` share the same underlying storage.
    *
    * @group pointwise_ops
    */
  def imag[D <: ComplexNN](input: Tensor[D]): Tensor[ComplexToReal[D]] =
    fromNative(torchNative.imag(input.native))

  /** Multiplies `input` by 2 ** `other`.
    *
    * @group pointwise_ops
    */
  def ldexp[D <: DType](input: Tensor[D], other: Tensor[D]): Tensor[D] =
    fromNative(torchNative.ldexp(input.native, other.native))

  /** Does a linear interpolation of two tensors `start` (given by `input`) and `end` (given by
    * `other`) based on a scalar or tensor weight and returns the resulting out tensor. out = start
    * + weight × (end − start)
    *
    * @group pointwise_ops
    */
  def lerp[D <: DType](
      input: Tensor[D],
      other: Tensor[D],
      weight: Tensor[D] | Float | Double
  ): Tensor[D] =
    fromNative(
      weight match
        case weight: Tensor[D] => torchNative.lerp(input.native, other.native, weight.native)
        case weight: Float     => torchNative.lerp(input.native, other.native, toScalar(weight))
        case weight: Double    => torchNative.lerp(input.native, other.native, toScalar(weight))
    )

  /** Computes the natural logarithm of the absolute value of the gamma function on `input`.
    *
    * @group pointwise_ops
    */
  def lgamma[D <: RealNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.lgamma(input.native))

  /** Returns a new tensor with the natural logarithm of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def log[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.log(input.native))

  /** Returns a new tensor with the logarithm to the base 10 of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def log10[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.log10(input.native))

  /** Returns a new tensor with the natural logarithm of (1 + input).
    *
    * @group pointwise_ops
    */
  def log1p[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.log1p(input.native))

  /** Returns a new tensor with the logarithm to the base 2 of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def log2[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.log2(input.native))

  /** Logarithm of the sum of exponentiations of the inputs. Calculates pointwise log `log(e**x +
    * e**y)`. This function is useful in statistics where the calculated probabilities of events may
    * be so small as to exceed the range of normal floating point numbers. In such cases the
    * logarithm of the calculated probability is stored. This function allows adding probabilities
    * stored in such a fashion. This op should be disambiguated with `torch.logsumexp()` which
    * performs a reduction on a single tensor.
    *
    * @group pointwise_ops
    */
  def logaddexp[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.logaddexp(input.native, other.native))

  /** Logarithm of the sum of exponentiations of the inputs in base-2. Calculates pointwise
    * `log2(2**x + 2**y)`. See torch.logaddexp() for more details.
    *
    * @group pointwise_ops
    */
  def logaddexp2[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.logaddexp2(input.native, other.native))

  /** Computes the element-wise logical AND of the given `input` tensors. Zeros are treated as False
    * and nonzeros are treated as True.
    *
    * @group pointwise_ops
    */
  def logicalAnd[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
    fromNative(torchNative.logical_and(input.native, other.native))

  /** Computes the element-wise logical NOT of the given `input` tensor. If the `input` tensor is
    * not a bool tensor, zeros are treated as False and non-zeros are treated as True.
    *
    * TODO If not specified, the output tensor will have the bool dtype.
    *
    * @group pointwise_ops
    */
  def logicalNot[D <: DType](input: Tensor[D]): Tensor[Bool] =
    fromNative(torchNative.logical_not(input.native))

  /** Computes the element-wise logical OR of the given `input` tensors. Zeros are treated as False
    * and nonzeros are treated as True.
    *
    * @group pointwise_ops
    */
  def logicalOr[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
    fromNative(torchNative.logical_or(input.native, other.native))

  /** Computes the element-wise logical XOR of the given `input` tensors. Zeros are treated as False
    * and nonzeros are treated as True.
    *
    * @group pointwise_ops
    */
  def logicalXor[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
    fromNative(torchNative.logical_xor(input.native, other.native))

  export torch.special.logit

  /** Given the legs of a right triangle, return its hypotenuse.
    *
    * @group pointwise_ops
    */
// TODO Change `D2 <: RealNN` once we fix property testing compilation
  def hypot[D <: RealNN, D2 <: FloatNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using AtLeastOneFloat[D, D2]): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.hypot(input.native, other.native))

  export torch.special.i0
  export torch.special.igamma
  export torch.special.igammac

  /** Multiplies input by other.
    *
    * @group pointwise_ops
    */
  def mul[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.mul(input.native, other.native))

  export torch.special.mvlgamma

  /** Replaces NaN, positive infinity, and negative infinity values in `input` with the values
    * specified by nan, posinf, and neginf, respectively. By default, NaNs are replaced with zero,
    * positive infinity is replaced with the greatest finite value representable by input’s dtype,
    * and negative infinity is replaced with the least finite value representable by input’s dtype.
    *
    * @group pointwise_ops
    */
  def nanToNum[D <: DType](
      input: Tensor[D],
      nan: Option[Double] = None,
      posinf: Option[Double] = None,
      neginf: Option[Double] = None
  ): Tensor[D] =
    fromNative(
      torchNative.nan_to_num(input.native, toOptional(nan), toOptional(posinf), toOptional(neginf))
    )

  /** Returns a new tensor with the negative of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def neg[D <: NumericNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.neg(input.native))

  /** Return the next floating-point value after `input` towards `other`, elementwise.
    *
    * @group pointwise_ops
    */
// TODO Change `D2 <: RealNN` once we fix property testing compilation
  def nextafter[D <: RealNN, D2 <: FloatNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using AtLeastOneFloat[D, D2]): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.nextafter(input.native, other.native))

  export torch.special.polygamma

  /** Returns input. Normally throws a runtime error if input is a bool tensor in pytorch.
    *
    * @group pointwise_ops
    */
  def positive[D <: NumericNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.positive(input.native))

  /** Takes the power of each element in `input` with exponent and returns a tensor with the result.
    * `exponent` can be either a single float number or a Tensor with the same number of elements as
    * input.
    *
    * @group pointwise_ops
    */
  def pow[D <: DType, D2 <: DType](input: Tensor[D], exponent: Tensor[D2])(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, D2] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, D2] NotEqual Complex32
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.pow(input.native, exponent.native))

  def pow[D <: DType, S <: ScalaType](input: Tensor[D], exponent: S)(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, ScalaToDType[S]] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, ScalaToDType[S]] NotEqual Complex32
  ): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(torchNative.pow(input.native, toScalar(exponent)))

  def pow[S <: ScalaType, D <: DType](input: S, exponent: Tensor[D])(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, ScalaToDType[S]] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, ScalaToDType[S]] NotEqual Complex32
  ): Tensor[Promoted[ScalaToDType[S], D]] =
    fromNative(torchNative.pow(toScalar(input), exponent.native))

// TODO Implement creation of QInts
// TODO quantized_batch_norm
// TODO quantized_max_pool1d
// TODO quantized_max_pool2d

  /** Returns a new tensor with each of the elements of `input` converted from angles in radians to
    * degrees.
    *
    * @group pointwise_ops
    */
  def rad2Deg[D <: RealNN | Bool](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.rad2deg(input.native))

  /** Returns a new tensor containing real values of the self tensor. The returned tensor and self
    * share the same underlying storage.
    *
    * @group pointwise_ops
    */
  def real[D <: DType](input: Tensor[D]): Tensor[ComplexToReal[D]] =
    fromNative(torchNative.real(input.native))

  /** Returns a new tensor with the reciprocal of the elements of `input`
    *
    * @group pointwise_ops
    */
  def reciprocal[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.reciprocal(input.native))

  /** Computes Python’s modulus operation entrywise. The result has the same sign as the divisor
    * `other` and its absolute value is less than that of `other`.
    *
    * @group pointwise_ops
    */
  def remainder[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.remainder(input.native, other.native))

  def remainder[D <: DType, R <: Real](
      input: Tensor[D],
      other: R
  ): Tensor[Promoted[D, ScalaToDType[R]]] =
    fromNative(torchNative.remainder(input.native, toScalar(other)))

  def remainder[D <: DType, R <: Real](
      input: R,
      other: Tensor[D]
  ): Tensor[Promoted[ScalaToDType[R], D]] =
    fromNative(torchNative.remainder(toScalar(input), other.native))

  /** Rounds elements of `input` to the nearest integer. If decimals is negative, it specifies the
    * number of positions to the left of the decimal point.
    *
    * @group pointwise_ops
    */
  def round[D <: FloatNN](input: Tensor[D], decimals: Long = 0): Tensor[D] =
    fromNative(torchNative.round(input.native, decimals))

  /** Returns a new tensor with the reciprocal of the square-root of each of the elements of
    * `input`.
    *
    * @group pointwise_ops
    */
  def rsqrt[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.rsqrt(input.native))

  export torch.special.sigmoid

  /** Returns a new tensor with the signs of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def sign[D <: RealNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.sign(input.native))

  /** This function is an extension of `torch.sign()` to complex tensors. It computes a new tensor
    * whose elements have the same angles as the corresponding elements of `input` and absolute
    * values (i.e. magnitudes) of one for complex tensors and is equivalent to torch.sign() for
    * non-complex tensors.
    *
    * @group pointwise_ops
    */
  def sgn[D <: DType](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.sgn(input.native))

  /** Tests if each element of `input`` has its sign bit set or not.
    *
    * @group pointwise_ops
    */
  def signbit[D <: RealNN](input: Tensor[D]): Tensor[Bool] =
    fromNative(torchNative.signbit(input.native))

  /** Returns a new tensor with the sine of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def sin[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.sin(input.native))

  export torch.special.sinc

  /** Returns a new tensor with the hyperbolic sine of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def sinh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.sinh(input.native))

  export torch.nn.functional.softmax

  /** Returns a new tensor with the square-root of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def sqrt[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.sqrt(input.native))

  /** Returns a new tensor with the square of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def square[D <: DType](input: Tensor[D]): Tensor[NumericPromoted[D]] =
    fromNative(torchNative.square(input.native))

  /** Subtracts `other`, scaled by `alpha`, from `input`.
    *
    * @group pointwise_ops
    */
  def sub[D <: NumericNN, D2 <: NumericNN](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.sub(input.native, other.native))

  def sub[D <: NumericNN, D2 <: NumericNN](
      input: Tensor[D],
      other: Tensor[D2],
      alpha: ScalaType
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.sub(input.native, other.native, toScalar(alpha)))

  def sub[D <: NumericNN, D2 <: NumericNN](
      input: Tensor[D],
      other: Numeric,
      alpha: ScalaType
  ): Tensor[Promoted[D, D2]] =
    fromNative(torchNative.sub(input.native, toScalar(other), toScalar(alpha)))

  /** Returns a new tensor with the tangent of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def tan[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.tan(input.native))

  /** Returns a new tensor with the hyperbolic tangent of the elements of `input`.
    *
    * @group pointwise_ops
    */
  def tanh[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.tanh(input.native))

  /** Alias for `torch.div()` with `rounding_mode=None`
    *
    * @group pointwise_ops
    */
  def trueDivide[D <: DType, D2 <: DType](
      input: Tensor[D],
      other: Tensor[D2]
  ): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.true_divide(input.native, other.native))

  def trueDivide[D <: DType, S <: ScalaType](
      input: Tensor[D],
      other: S
  ): Tensor[FloatPromoted[Promoted[D, ScalaToDType[S]]]] =
    fromNative(torchNative.true_divide(input.native, toScalar(other)))

  /** Returns a new tensor with the truncated integer values of the elements of `input`
    *
    * @group pointwise_ops
    */
  def trunc[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
    fromNative(torchNative.trunc(input.native))

  export torch.special.xlogy
}
