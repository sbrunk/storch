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
package pooling

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{MaxPool2dImpl, MaxPool2dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class MaxPool2d[D <: BFloat16 | Float32 | Float64: Default](
    kernelSize: Int | (Int, Int),
    stride: Option[Int | (Int, Int)] = None,
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    // returnIndices: Boolean = false,
    ceilMode: Boolean = false
) extends TensorModule[D]:

  private val options: MaxPool2dOptions = MaxPool2dOptions(toNative(kernelSize))
  stride.foreach(s => options.stride().put(toNative(s)))
  options.padding().put(toNative(padding))
  options.dilation().put(toNative(dilation))
  options.ceil_mode().put(ceilMode)

  override private[torch] val nativeModule: MaxPool2dImpl = MaxPool2dImpl(options)

  override def hasBias(): Boolean = false

  override def toString(): String =
    s"MaxPool2d(kernelSize=$kernelSize, stride=$stride, padding=$padding, dilation=$dilation, ceilMode=$ceilMode)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  // TODO forward_with_indices
