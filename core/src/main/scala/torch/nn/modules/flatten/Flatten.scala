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
package modules
package flatten

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{FlattenImpl, FlattenOptions}
import torch.internal.NativeConverters.fromNative

// format: off
/** Flattens a contiguous range of dims into a tensor. For use with [[nn.Sequential]].
  *
  * Shape:
  * \- Input: $(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)$,' where $S_{i}$ is the size
  * at dimension $i$ and $*$ means any number of dimensions including none.
  * \- Output: $(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)$.
  *
  * Example:
  *
  * ```scala
  * import torch.nn
  *
  * val input = torch.randn(Seq(32, 1, 5, 5))
  * // With default parameters
  * val m1 = nn.Flatten()
  * // With non-default parameters
  * val m2 = nn.Flatten(0, 2)
  * ```
  *
  * @group nn_flatten
  *
  * @param startDim
  *   first dim to flatten
  * @param endDim
  *   last dim to flatten
  */
// format: on
final class Flatten[D <: DType: Default](startDim: Int = 1, endDim: Int = -1)
    extends TensorModule[D]:

  private val options = FlattenOptions()
  options.start_dim().put(startDim)
  options.end_dim().put(endDim)

  override val nativeModule: FlattenImpl = FlattenImpl(options)

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  override def toString = getClass().getSimpleName()
