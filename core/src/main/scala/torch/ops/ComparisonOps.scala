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

import org.bytedeco.pytorch.global.torch as torchNative
import internal.NativeConverters.fromNative

/** Comparison Ops
  *
  * https://pytorch.org/docs/stable/torch.html#comparison-ops
  */
private[torch] trait ComparisonOps {

  def allclose(
      input: Tensor[?],
      other: Tensor[?],
      rtol: Double = 1e-05,
      atol: Double = 1e-08,
      equalNan: Boolean = false
  ): Boolean =
    torchNative.allclose(input.native, other.native, rtol, atol, equalNan)

  /** Returns the indices that sort a tensor along a given dimension in ascending order by value.
    *
    * This is the second value returned by `torch.sort`. See its documentation for the exact
    * semantics of this method.
    *
    * If `stable` is `True` then the sorting routine becomes stable, preserving the order of
    * equivalent elements. If `False`, the relative order of values which compare equal is not
    * guaranteed. `True` is slower.
    *
    * Args: {input} dim (int, optional): the dimension to sort along descending (bool, optional):
    * controls the sorting order (ascending or descending) stable (bool, optional): controls the
    * relative order of equivalent elements
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.randn(Seq(4, 4))
    * // tensor dtype=float32, shape=[4, 4], device=CPU
    * // [[ 0.0785,  1.5267, -0.8521,  0.4065],
    * //  [ 0.1598,  0.0788, -0.0745, -1.2700],
    * //  [ 1.2208,  1.0722, -0.7064,  1.2564],
    * //  [ 0.0669, -0.2318, -0.8229, -0.9280]]
    *
    * torch.argsort(a, dim = 1)
    * // tensor dtype=int64, shape=[4, 4], device=CPU
    * // [[2, 0, 3, 1],
    * //  [3, 2, 1, 0],
    * //  [2, 1, 0, 3],
    * //  [3, 2, 1, 0]]
    * ```
    *
    * @group comparison_ops
    */
  def argsort[D <: RealNN](
      input: Tensor[D],
      dim: Int = -1,
      descending: Boolean = false
      // TODO implement stable, there are two boolean args in argsort and are not in order
      // stable: Boolean = false
  ): Tensor[Int64] =
    fromNative(
      torchNative.argsort(input.native, dim.toLong, descending)
    )
}
