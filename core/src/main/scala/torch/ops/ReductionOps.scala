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

/** Reduction Ops
  *
  * https://pytorch.org/docs/stable/torch.html#reduction-ops
  */

// TODO argmax Returns the indices of the maximum value of all elements in the `input` tensor.
// TODO argmin Returns the indices of the minimum value(s) of the flattened tensor or along a dimension
// TODO amax Returns the maximum value of each slice of the `input` tensor in the given dimension(s) dim.
// TODO amin Returns the minimum value of each slice of the `input` tensor in the given dimension(s) dim.
// TODO aminmax Computes the minimum and maximum values of the `input` tensor.
// TODO all Tests if all elements in `input` evaluate to True.
// TODO any Tests if any element in `input` evaluates to True.
// TODO max Returns the maximum value of all elements in the `input` tensor.
// TODO min Returns the minimum value of all elements in the `input` tensor.
// TODO dist Returns the p-norm of (input - other)
// TODO logsumexp Returns the log of summed exponentials of each row of the `input` tensor in the given dimension dim.
// TODO mean Returns the mean value of all elements in the `input` tensor.
// TODO nanmean Computes the mean of all non-NaN elements along the specified dimensions.
// TODO median Returns the median of the values in input.
// TODO nanmedian Returns the median of the values in input, ignoring NaN values.
// TODO mode Returns a namedtuple (values, indices) where values is the mode value of each row of the `input` tensor in the given dimension dim, i.e. a value which appears most often in that row, and indices is the index location of each mode value found.
// TODO norm Returns the matrix norm or vector norm of a given tensor.
// TODO nansum Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.
// TODO prod Returns the product of all elements in the `input` tensor.
// TODO quantile Computes the q-th quantiles of each row of the `input` tensor along the dimension dim.
// TODO nanquantile This is a variant of torch.quantile() that "ignores" NaN values, computing the quantiles q as if NaN values in `input` did not exist.
// TODO std Calculates the standard deviation over the dimensions specified by dim.
// TODO std_mean Calculates the standard deviation and mean over the dimensions specified by dim.

/* Returns the sum of all elements in the `input` tensor. */
def sum[D <: DType](
    input: Tensor[D],
    dim: Array[Long] = Array(),
    keepdim: Boolean = false,
    dtype: Option[DType] = None
): Tensor[D] =
  val lar = new org.bytedeco.pytorch.LongArrayRef(dim, dim.size)
  val laro = new org.bytedeco.pytorch.LongArrayRefOptional(lar)
  // TODO Add dtype
  val sto = new org.bytedeco.pytorch.ScalarTypeOptional()
  Tensor(torchNative.sum(input.native, dim, keepdim, sto))

// TODO unique Returns the unique elements of the `input` tensor.
// TODO unique_consecutive Eliminates all but the first element from every consecutive group of equivalent elements.

/* TODO Calculates the variance over the dimensions specified by dim. */
//def variance[D <: DType](input: Tensor[D], dim: Seq[Int] = Nil, correction: Option[Int] = None, keepdim: Boolean = false) =
//  Tensor(torchNative.`var`(input.native, dim.toArray.map(_.toLong), toOptional(correction), keepdim))

// TODO var_mean Calculates the variance and mean over the dimensions specified by dim.
// TODO count_nonzero Counts the number of non-zero values in the tensor `input` along the given dim.
