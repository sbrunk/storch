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
package activation

import org.bytedeco.pytorch
import org.bytedeco.pytorch.LogSoftmaxImpl
import org.bytedeco.pytorch.LogSoftmaxOptions
import torch.internal.NativeConverters.fromNative

/** Applies the log(Softmax(x)) function to an n-dimensional input Tensor. The LogSoftmax
  * formulation can be simplified as:
  *
  * TODO LaTeX
  *
  * Example:
  *
  * ```scala sc
  * import torch.*
  * val m = nn.LogSoftmax(dim = 1)
  * val input = torch.randn(Seq(2, 3))
  * val output = m(input)
  * ```
  */
final class LogSoftmax[D <: DType: Default](dim: Int) extends TensorModule[D]:
  private val options = new LogSoftmaxOptions(dim)
  options.dim().put(dim)

  override val nativeModule: LogSoftmaxImpl = LogSoftmaxImpl(options)

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
