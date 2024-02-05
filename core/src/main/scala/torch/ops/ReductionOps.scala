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
import org.bytedeco.pytorch.ScalarTypeOptional

/** Reduction Ops
  *
  * https://pytorch.org/docs/stable/torch.html#reduction-ops
  *
  * @define {single_keepdim_details}
  *   If `keepdim` is `true`, the output tensor is of the same size as `input` except in the
  *   dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see `torch.squeeze`),
  *   resulting in the output tensor having 1 fewer dimension than `input`.
  *
  * @define {multi_keepdim_details}
  *   If `keepdim` is `true`, the output tensor is of the same size as `input` except in the
  *   dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed (see `torch.squeeze`),
  *   resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).
  *
  * @define {reduceops_dtype}
  *   the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`
  *   before the operation is performed. This is useful for preventing data type overflows.
  */
private[torch] trait ReductionOps {

  /** Returns the indices of the maximum value of all elements in the tensor.
    *
    * This is the second value returned by torch.max(). See its documentation for the exact
    * semantics of this method.
    *
    * Example:
    * ```scala sc
    * val a = torch.rand(Seq(1, 3))
    * torch.argmax(a)
    * // tensor dtype=int64, shape=[], device=CPU
    * // 1
    * ```
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce. If [[None]], the argmin of the flattened input is returned.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @return
    */
  def argmax[D <: IntNN | FloatNN](
      input: Tensor[D],
      dim: Int | Option[Int] = None,
      keepdim: Boolean = false
  ): Tensor[Int64] = fromNative(
    torchNative.argmax(input.native, dim.toOptional, keepdim)
  )

  /** Returns the indices of the minimum value of all elements in the tensor.
    *
    * This is the second value returned by torch.min(). See its documentation for the exact
    * semantics of this method.
    *
    * Example:
    * ```scala sc
    * val a = torch.rand(Seq(1, 3))
    * torch.argmin(a)
    * // tensor dtype=int64, shape=[], device=CPU
    * // 1
    * ```
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce. If [[None]], the argmin of the flattened input is returned.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @return
    */
  def argmin[D <: IntNN | FloatNN](
      input: Tensor[D],
      dim: Int | Option[Int] = None,
      keepdim: Boolean = false
  ): Tensor[Int64] = fromNative(
    torchNative.argmin(input.native, dim.toOptional, keepdim)
  )

  /** Returns the maximum value of each slice of the `input` tensor in the given dimension(s) `dim`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @return
    */
  def amax[D <: RealNN](
      input: Tensor[D],
      dim: Int | Seq[Int],
      keepdim: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.amax(input.native, dim.toArray, keepdim)
    )

  /** Returns the minimum value of each slice of the `input` tensor in the given dimension(s) `dim`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @return
    */
  def amin[D <: RealNN](
      input: Tensor[D],
      dim: Int | Seq[Int],
      keepdim: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.amin(input.native, dim.toArray, keepdim)
    )

  /** Computes the minimum and maximum values of the `input` tensor.
    *
    * @group reduction_ops
    *
    * @param dim
    *   The dimension along which to compute the values. If [[None]], computes the values over the
    *   entire input tensor. Default is [[None]].
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @return
    */
  def aminmax[D <: RealNN](
      input: Tensor[D],
      dim: Int | Option[Int] = None,
      keepdim: Boolean = false
  ): (Tensor[D], Tensor[D]) =
    val native = torchNative.aminmax(input.native, dim.toOptional, keepdim)
    (fromNative(native.get0()), fromNative(native.get1()))

  /** Tests if all elements of this tensor evaluate to `true`.
    *
    * @group reduction_ops
    */
  def all[D <: DType](input: Tensor[D]): Tensor[Bool] = fromNative(torchNative.all(input.native))

  /** For each row of `input` in the given dimension `dim`, returns `true` if all elements in the
    * row evaluate to `true` and `false` otherwise.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    */
  def all[D <: DType](input: Tensor[D], dim: Int, keepdim: Boolean = false): Tensor[Bool] =
    fromNative(
      torchNative.all(input.native, dim, keepdim)
    )

  /** Tests if any elements of this tensor evaluate to `true`.
    *
    * @group reduction_ops
    */
  def any[D <: DType](input: Tensor[D]): Tensor[Bool] = fromNative(torchNative.any(input.native))

  /** For each row of `input` in the given dimension `dim`, returns `true` if any element in the row
    * evaluates to `true` and `false` otherwise.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    */
  def any[D <: DType](input: Tensor[D], dim: Int, keepdim: Boolean = false): Tensor[Bool] =
    fromNative(
      torchNative.any(input.native, dim, keepdim)
    )

  /** Returns the maximum value of all elements in the `input` tensor.
    *
    * @group reduction_ops
    */
  def max[D <: RealNN](input: Tensor[D]): Tensor[Int64] = fromNative(input.native.max())

  /** Returns a [[TensorTuple]] `(values, indices)` where `values` is the maximum value of each row
    * of the `input` tensor in the given dimension `dim`. And `indices` is the index location of
    * each maximum value found (argmax).
    *
    * ${single_keepdim_details}
    *
    * @note
    *   If there are multiple maximal values in a reduced row then the indices of the first maximal
    *   value are returned.
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    */
  def max[D <: RealNN](input: Tensor[D], dim: Int, keepdim: Boolean = false): TensorTuple[D] =
    val nativeTuple = torchNative.max(input.native, dim, keepdim)
    TensorTuple(
      values = fromNative[D](nativeTuple.get0),
      indices = new Int64Tensor(nativeTuple.get1)
    )

  /** Returns the maximum value of all elements in the `input` tensor.
    *
    * @group reduction_ops
    */
  def min[D <: RealNN](input: Tensor[D]): Tensor[Int64] = fromNative(input.native.min())

  /** Returns a [[TensorTuple]] `(values, indices)` where `values` is the minimum value of each row
    * of the `input` tensor in the given dimension `dim`. And `indices` is the index location of
    * each maximum value found (argmax).
    *
    * ${single_keepdim_details}
    *
    * @note
    *   If there are multiple minimal values in a reduced row then the indices of the first minimal
    *   value are returned.
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    */
  def min[D <: RealNN](input: Tensor[D], dim: Int, keepdim: Boolean = false): TensorTuple[D] =
    val nativeTuple = torchNative.min(input.native, dim, keepdim)
    TensorTuple(
      values = fromNative[D](nativeTuple.get0),
      indices = new Int64Tensor(nativeTuple.get1)
    )

  /** Returns the p-norm of (`input` - `other`)
    *
    * The shapes of `input` and `other` must be broadcastable.
    *
    * @group reduction_ops
    *
    * @param p
    *   the norm to be computed
    */
  def dist[D <: NumericNN, D2 <: NumericNN](
      input: Tensor[D],
      other: Tensor[D2],
      p: Float = 2
  )(using
      AtLeastOneFloat[D, D2]
  ): Tensor[Promoted[FloatPromoted[ComplexToReal[D]], FloatPromoted[ComplexToReal[D2]]]] =
    fromNative(torchNative.dist(input.native, other.native, toScalar(p)))

  /** Returns the log of summed exponentials of each row of the `input` tensor in the given
    * dimension `dim`. The computation is numerically stabilized.
    *
    * For summation index $j$ given by `dim` and other indices $i$, the result is
    *
    * > $$\text{{logsumexp}}(x)_{{i}} = \log \sum_j \exp(x_{{ij}})$$
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    */
  def logsumexp[D <: RealNN](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false
  ): Tensor[D] = fromNative(
    torchNative.logsumexp(input.native, dim.toArray, keepdim)
  )

  /** Returns the mean value of each row of the `input` tensor in the given dimension `dim`. If
    * `dim` is a list of dimensions, reduce over all of them.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def mean[D <: FloatNN | ComplexNN, D2 <: FloatNN | ComplexNN | Derive](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    // TODO factor out
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    fromNative(
      torchNative.mean(
        input.native,
        dim.toArray,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

  /** Computes the mean of all [non-NaN] elements along the specified dimensions.
    *
    * This function is identical to `torch.mean` when there are no [NaN] values in the `input`
    * tensor. In the presence of [NaN], `torch.mean` will propagate the [NaN] to the output whereas
    * `torch.nanmean` will ignore the [NaN] values ([torch.nanmean(a)] is equivalent to
    * [torch.mean(a\[\~a.isnan()\])]).
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def nanmean[D <: FloatNN, D2 <: DType | Derive](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    // TODO factor out
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    fromNative(
      torchNative.nanmean(
        input.native,
        dim.toArray,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

    /** Returns the median of the values in `input`.
      *
      * @note
      *   The median is not unique for `input` tensors with an even number of elements. In this case
      *   the lower of the two medians is returned. To compute the mean of both medians, use
      *   `torch.quantile` with `q=0.5` instead.
      *
      * Warning
      *
      * This function produces deterministic (sub)gradients unlike `median(dim=0)`
      *
      * @group reduction_ops
      */
  def median[D <: NumericRealNN](
      input: Tensor[D]
  ): Tensor[D] = fromNative(torchNative.median(input.native))

  /** Returns a [[TensorTuple]] `(values, indices)` where `values` contains the median of each row
    * of `input` in the dimension `dim`, and `indices` contains the index of the median values found
    * in the dimension `dim`.
    *
    * By default, `dim` is the last dimension of the `input` tensor.
    *
    * ${single_keepdim_details}
    *
    * @note
    *   The median is not unique for `input` tensors with an even number of elements in the
    *   dimension `dim`. In this case the lower of the two medians is returned. To compute the mean
    *   of both medians in `input`, use `torch.quantile` with `q=0.5` instead.
    *
    * Warning
    *
    * `indices` does not necessarily contain the first occurrence of each median value found, unless
    * it is unique. The exact implementation details are device-specific. Do not expect the same
    * result when run on CPU and GPU in general. For the same reason do not expect the gradients to
    * be deterministic.
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def median[D <: NumericRealNN, D2 <: DType | Derive](
      input: Tensor[D],
      dim: Int = -1,
      keepdim: Boolean = false
  ): TensorTuple[D] =
    val nativeTuple = torchNative.median(input.native, dim, keepdim)
    TensorTuple(
      values = fromNative[D](nativeTuple.get0),
      indices = new Int64Tensor(nativeTuple.get1)
    )

    /** Returns the median of the values in `input`, ignoring `NaN` values.
      *
      * This function is identical to `torch.median` when there are no `NaN` values in `input`. When
      * `input` has one or more `NaN` values, `torch.median` will always return `NaN`, while this
      * function will return the median of the non-`NaN` elements in `input`. If all the elements in
      * `input` are `NaN` it will also return `NaN`.
      *
      * @group reduction_ops
      */
  def nanmedian[D <: NumericRealNN](
      input: Tensor[D]
  ): Tensor[D] = fromNative(torchNative.nanmedian(input.native))

  /** Returns a [[TensorTuple]] ``(values, indices)`` where ``values`` contains the median of each
    * row of `input` in the dimension `dim`, ignoring ``NaN`` values, and ``indices`` contains the
    * index of the median values found in the dimension `dim`.
    *
    * This function is identical to :func:`torch.median` when there are no ``NaN`` values in a
    * reduced row. When a reduced row has one or more ``NaN`` values, :func:`torch.median` will
    * always reduce it to ``NaN``, while this function will reduce it to the median of the
    * non-``NaN`` elements. If all the elements in a reduced row are ``NaN`` then it will be reduced
    * to ``NaN``, too.
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def nanmedian[D <: NumericRealNN, D2 <: DType | Derive](
      input: Tensor[D],
      dim: Int = -1,
      keepdim: Boolean = false
  ): TensorTuple[D] =
    val nativeTuple = torchNative.nanmedian(input.native, dim, keepdim)
    TensorTuple(
      values = fromNative[D](nativeTuple.get0),
      indices = new Int64Tensor(nativeTuple.get1)
    )

  /** Returns a [[TensorTuple]] `(values, indices)` where `values` is the mode value of each row of
    * the `input` tensor in the given dimension `dim`,
    * i.e. a value which appears most often in that row, and `indices` is the index location of each
    * mode value found.
    *
    * By default, `dim` is the last dimension of the `input` tensor.
    *
    * ${single_keepdim_details}
    *
    * @note
    *   This function is not defined for `torch.cuda.Tensor` yet.
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    */
  def mode[D <: RealNN](
      input: Tensor[D],
      dim: Int = -1,
      keepdim: Boolean = false
  ): TensorTuple[D] =
    val nativeTuple = torchNative.mode(input.native, dim, keepdim)
    TensorTuple(
      values = fromNative[D](nativeTuple.get0),
      indices = new Int64Tensor(nativeTuple.get1)
    )

  /** Returns the sum of each row of the `input` tensor in the given dimension `dim`, treating Not a
    * Numbers (NaNs) as zero. If `dim` is a list of dimensions, reduce over all of them.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def nansum[D <: RealNN, D2 <: DType | Derive](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    // TODO factor out
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    fromNative(
      torchNative.nansum(
        input.native,
        dim.toArray,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

  /** Returns the product of all elements in the `input` tensor.
    *
    * @group reduction_ops
    */
  def prod[D <: DType, D2 <: DType | Derive](
      input: Tensor[D]
  ): Tensor[D] = fromNative(torchNative.prod(input.native))

  /** Returns the product of all elements in the `input` tensor.
    *
    * @group reduction_ops
    *
    * @param dtype
    *   ${reduceops_dtype}
    */
  def prod[D <: DType](
      input: Tensor[?],
      dtype: D
  ): Tensor[D] = fromNative(
    torchNative.prod(input.native, new ScalarTypeOptional(dtype.toScalarType))
  )

  /** Returns the product of each row of the `input` tensor in the given dimension `dim`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension to reduce.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def prod[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dim: Int,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    // TODO factor out
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    fromNative(
      torchNative.prod(
        input.native,
        dim,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

    /** Computes the q-th quantiles of each row of the `input` tensor along the dimension `dim`.
      *
      * To compute the quantile, we map q in \[0, 1\] to the range of indices \[0, n\] to find the
      * location of the quantile in the sorted input. If the quantile lies between two data points
      * `a < b` with indices `i` and `j` in the sorted order, result is computed according to the
      * given `interpolation` method as follows:
      *
      *   - `linear`: `a + (b - a) * fraction`, where `fraction` is the fractional part of the
      *     computed quantile index.
      *   - `lower`: `a`.
      *   - `higher`: `b`.
      *   - `nearest`: `a` or `b`, whichever\'s index is closer to the computed quantile index
      *     (rounding down for .5 fractions).
      *   - `midpoint`: `(a + b) / 2`.
      *
      * If `q` is a 1D tensor, the first dimension of the output represents the quantiles and has
      * size equal to the size of `q`, the remaining dimensions are what remains from the reduction.
      *
      * @note
      *   By default `dim` is `None` resulting in the `input` tensor being flattened before
      *   computation.
      *
      * @group reduction_ops
      *
      * @param q
      *   (float or Tensor): a scalar or 1D tensor of values in the range [0, 1].
      * @param dim
      *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
      * @param keepdim
      *   whether the output tensor has `dim` retained or not.
      * @param interpolation
      *   interpolation method to use when the desired quantile lies between two data points. Can be
      *   ``linear``, ``lower``, ``higher``, ``midpoint`` and ``nearest``. Default is ``linear``.
      */
// def quantile[D <: DType, D2 <: DType | Derive](
//     input: Tensor[D],
//     q: Double | Tensor[?], // TODO only float tensor?
//     dim: Option[Int] = None,
//     keepdim: Boolean = false,

// ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
//   fromNative(
//     torchNative.quantile(
//       input.native,
//       q,
//       dim.toOptional,
//       keepdim,
//       // TODO figure out how to create c10:string_view values
//     )
//   )

// TODO nanquantile This is a variant of torch.quantile() that "ignores" NaN values, computing the quantiles q as if NaN values in `input` did not exist.
// (same issue as quantile, need to figure out how to create c10:string_view values)

  /** Calculates the standard deviation over the dimensions specified by `dim`. `dim` can be a
    * single dimension, list of dimensions, or `None` to reduce over all dimensions.
    *
    * The standard deviation ($\sigma$) is calculated as
    *
    * $$\sigma = \sqrt{\frac{1}{N - \delta N}\sum_{i=0}^{N-1}(x_i-\bar{x})^2}$$
    *
    * where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of
    * samples and $\delta N$ is the `correction`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param correction
    *   difference between the sample size and sample degrees of freedom. Defaults to [Bessel\'s
    *   correction](https://en.wikipedia.org/wiki/Bessel%27s_correction), `correction=1`.
    */
  def std[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      correction: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.std(
        input.native,
        dim.toArray,
        correction.toScalarOptional,
        keepdim
      )
    )

  /** Calculates the standard deviation and mean over the dimensions specified by `dim`. `dim` can
    * be a single dimension, list of dimensions, or `None` to reduce over all dimensions.
    *
    * The standard deviation ($\sigma$) is calculated as
    *
    * $$\sigma = \sqrt{\frac{1}{N - \delta N}\sum_{i=0}^{N-1}(x_i-\bar{x})^2}$$
    *
    * where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of
    * samples and $\delta N$ is the `correction`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param correction
    *   difference between the sample size and sample degrees of freedom. Defaults to [Bessel\'s
    *   correction](https://en.wikipedia.org/wiki/Bessel%27s_correction), `correction=1`.
    * @return
    *   A tuple (std, mean) containing the standard deviation and mean.
    */
  def stdMean[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      correction: Int = 1
  ): (Tensor[D], Tensor[D]) =
    val nativeTuple =
      torchNative.std_mean(
        input.native,
        dim.toArray,
        correction.toScalarOptional,
        keepdim
      )
    (fromNative[D](nativeTuple.get0), fromNative[D](nativeTuple.get1))

  /** Returns the sum of all elements in the `input` tensor.
    *
    * @group reduction_ops
    */
  def sum[D <: DType, D2 <: DType | Derive](
      input: Tensor[D]
  ): Tensor[D] = fromNative(torchNative.sum(input.native))

  /** Returns the sum of all elements in the `input` tensor.
    *
    * @group reduction_ops
    *
    * @param dtype
    *   ${reduceops_dtype}
    */
  def sum[D <: DType](
      input: Tensor[?],
      dtype: D
  ): Tensor[D] = fromNative(
    torchNative.sum(input.native, new ScalarTypeOptional(dtype.toScalarType))
  )

  /** Returns the sum of each row of the `input` tensor in the given dimension `dim`.
    *
    * If dim is a list of dimensions, reduce over all of them.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param dtype
    *   ${reduceops_dtype}
    */
  def sum[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    // TODO factor out
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    fromNative(
      torchNative.sum(
        input.native,
        dim.toArray,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

  /** Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices `input`, the
    * other elements of the result tensor `out` are set to 0.
    *
    * The lower triangular part of the matrix is defined as the elements on and below the diagonal.
    *
    * The argument [[torch.diagonal diagonal]] controls which diagonal to consider. If `diagonal =
    * 0`, all elements on and below the main diagonal are retained. A positive value includes just
    * as many diagonals above the main diagonal, and similarly a negative value excludes just as
    * many diagonals below the main diagonal. The main diagonal are the set of indices ${(i,i)}$ for
    * $i \in [0,min⁡{d1,d2}−1]$ where $d_1​,d_2​$ are the dimensions of the matrix.
    *
    * @example
    *   {{{
    * val a = torch.randn(3, 3)
    * torch.tril(a)
    * val b = torch.randn(4, 6)
    * torch.tril(b, diagonal=1)
    * torch.tril(b, diagonal=-1)
    *   }}}
    *
    * @param input
    *   the input tensor.
    * @param diagonal
    *   the diagonal to consider
    * @return
    *   out the output tensor.
    */
  def tril[D <: DType](input: Tensor[D], diagonal: Int = 0): Tensor[D] =
    fromNative(torchNative.tril(input.native, diagonal.toLong))

    // TODO unique Returns the unique elements of the `input` tensor.
    // seems to be implemented in https://github.com/pytorch/pytorch/blob/main/torch/functional.py
    // and calls different native functions depending on dim unique_dim or _unique2

    // TODO unique_consecutive Eliminates all but the first element from every consecutive group of equivalent elements.
    // Similar to unique we should look at  _unique_consecutive_impl in Python first
    // https://github.com/pytorch/pytorch/blob/dbc8eb2a8fd894fbc110bbb9f70037249868afa8/torch/functional.py#L827

    // TODO var
    /* TODO Calculates the variance over the dimensions specified by dim. */
    // def variance[D <: DType](input: Tensor[D], dim: Seq[Int] = Nil, correction: Option[Int] = None, keepdim: Boolean = false) =
    //  fromNative(torchNative.`var`(input.native, dim.toArray.map(_.toLong), toOptional(correction), keepdim))

  /** Calculates the variance over the dimensions specified by `dim`. `dim` can be a single
    * dimension, list of dimensions, or `None` to reduce over all dimensions.
    *
    * The variance ($\sigma^2$) is calculated as
    *
    * $$\sigma^2 = \frac{1}{N - \delta N}\sum_{i=0}^{N-1}(x_i-\bar{x})^2$$
    *
    * where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of
    * samples and $\delta N$ is the `correction`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param correction
    *   difference between the sample size and sample degrees of freedom. Defaults to [Bessel\'s
    *   correction](https://en.wikipedia.org/wiki/Bessel%27s_correction), `correction=1`.
    */
  def variance[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      correction: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.`var`(
        input.native,
        dim.toArray,
        correction.toScalarOptional,
        keepdim
      )
    )

  /** Calculates the variance and mean over the dimensions specified by `dim`. `dim` can be a single
    * dimension, list of dimensions, or `None` to reduce over all dimensions.
    *
    * The variance ($\sigma^2$) is calculated as
    *
    * $$\sigma^2 = \frac{1}{N - \delta N}\sum_{i=0}^{N-1}(x_i-\bar{x})^2$$
    *
    * where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of
    * samples and $\delta N$ is the `correction`.
    *
    * ${single_keepdim_details}
    *
    * @group reduction_ops
    *
    * @param dim
    *   the dimension or dimensions to reduce. If empty, all dimensions are reduced.
    * @param keepdim
    *   whether the output tensor has `dim` retained or not.
    * @param correction
    *   difference between the sample size and sample degrees of freedom. Defaults to [Bessel\'s
    *   correction](https://en.wikipedia.org/wiki/Bessel%27s_correction), `correction=1`.
    * @return
    *   A tuple (var, mean) containing the variance and mean.
    */
  def varMean[D <: FloatNN | ComplexNN](
      input: Tensor[D],
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      correction: Int = 1
  ): (Tensor[D], Tensor[D]) =
    val nativeTuple =
      torchNative.var_mean(
        input.native,
        dim.toArray,
        correction.toScalarOptional,
        keepdim
      )
    (fromNative[D](nativeTuple.get0), fromNative[D](nativeTuple.get1))

  /** Counts the number of non-zero values in the tensor `input` along the given `dim`. If no dim is
    * specified then all non-zeros in the tensor are counted.
    *
    * @group reduction_ops
    *
    * @param dim
    *   Dim or seq of dims along which to count non-zeros.
    */
  def countNonzero(
      input: Tensor[?],
      dim: Int | Seq[Int] = Seq.empty
  ): Tensor[Int64] =
    val nativeDim = dim.toArray
    fromNative(
      if nativeDim.isEmpty then torchNative.count_nonzero(input.native)
      else torchNative.count_nonzero(input.native, nativeDim: _*)
    )
}
