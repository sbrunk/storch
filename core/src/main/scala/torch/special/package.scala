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

import org.bytedeco.pytorch.global.torch as torchNative

import internal.NativeConverters.*

package object special:
  /** Computes the logarithmic derivative of the gamma function on `input`. */
  def digamma[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.digamma(input.native))

  /** Computes the error function of `input`. */
  def erf[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.erf(input.native))

  /** Computes the complementary error function of `input`. */
  def erfc[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.erfc(input.native))

  /** Computes the inverse error function of `input`. The inverse error function is defined in the
    * range (−1,1)
    */
  def erfinv[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.erfinv(input.native))

  /** Computes the base two exponential function of `input`. */
  def exp2[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.exp2(input.native))

  /** Computes the exponential of the elements minus 1 of `input`. */
  def expm1[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.expm1(input.native))

  /** Computes the zeroth order modified Bessel function of the first kind for each element of
    * `input`.
    */
  def i0[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.i0(input.native))

  /** Computes the regularized lower incomplete gamma function */
  // NOTE it is named `gammainc` in pytorch torch.special
  // TODO Change `D2 <: RealNN` once we fix property testing compilation
  def igamma[D <: RealNN, D2 <: FloatNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using AtLeastOneFloat[D, D2]): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.igamma(input.native, other.native))

  /** Computes the regularized upper incomplete gamma function */
  // NOTE it is named `gamaincc` in pytorch torch.special
  // TODO Change `D2 <: RealNN` once we fix property testing compilation
  def igammac[D <: RealNN, D2 <: FloatNN](
      input: Tensor[D],
      other: Tensor[D2]
  )(using AtLeastOneFloat[D, D2]): Tensor[FloatPromoted[Promoted[D, D2]]] =
    fromNative(torchNative.igammac(input.native, other.native))

  /** Returns a new tensor with the logit of the elements of `input`. `input` is clamped to [eps, 1
    * \- eps] when eps is not None. When eps is None and input < 0 or input > 1, the function will
    * yields NaN.
    */
  def logit[D <: RealNN](input: Tensor[D], eps: Option[Double]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.logit(input.native, toOptional(eps)))

  /** Computes the multivariate log-gamma function with dimension p element-wise */
  // NOTE it is named `multigammaln` in pytorch torch.special
  def mvlgamma[D <: NumericRealNN](input: Tensor[D], p: Int): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.mvlgamma(input.native, p))

  /** Computes the nth derivative of the digamma function on `input`. n≥0 is called the order of the
    * polygamma function.
    */
  def polygamma[D <: RealNN](n: Int, input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.polygamma(n, input.native))

  /** Computes the expit (also known as the logistic sigmoid function) of the elements of `input`.
    */
  // NOTE it is named `expit` in pytorch torch.special
  def sigmoid[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.sigmoid(input.native))

  /** Returns a new tensor with the normalized sinc of the elements of `input`. */
  def sinc[D <: DType](input: Tensor[D]): Tensor[FloatPromoted[D]] =
    fromNative(torchNative.sinc(input.native))

  /** Computes `input * log(other)` with the following cases. */
  // TODO handle Scalar `input`
  def xlogy[D <: RealNN, D2 <: RealNN](
      input: Tensor[D],
      other: TensorOrReal[D2]
  ): Tensor[FloatPromoted[D]] =
    fromNative(
      other match
        case other: Tensor[D2] =>
          torchNative.xlogy(input.native, other.native)
        case other: Real =>
          torchNative.xlogy(input.native, toScalar(other))
    )
