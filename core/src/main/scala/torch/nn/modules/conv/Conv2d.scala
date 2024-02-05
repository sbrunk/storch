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
package conv

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{Conv2dImpl, Conv2dOptions, kZeros, kReflect, kReplicate, kCircular}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.conv.Conv2d.PaddingMode

/** Applies a 2D convolution over an input signal composed of several input planes.
  *
  * @group nn_conv
  */
final class Conv2d[ParamType <: FloatNN | ComplexNN: Default](
    inChannels: Long,
    outChannels: Long,
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int) = 1,
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    groups: Int = 1,
    bias: Boolean = true,
    paddingMode: PaddingMode = PaddingMode.Zeros
) extends HasParams[ParamType]
    with TensorModule[ParamType]:

  private val options = new Conv2dOptions(inChannels, outChannels, toNative(kernelSize))
  options.stride().put(toNative(stride))
  options.padding().put(toNative(padding))
  options.dilation().put(toNative(dilation))
  options.groups().put(groups)
  options.bias().put(bias)
  private val paddingModeNative = paddingMode match
    case PaddingMode.Zeros     => new kZeros
    case PaddingMode.Reflect   => new kReflect
    case PaddingMode.Replicate => new kReplicate
    case PaddingMode.Circular  => new kCircular
  options.padding_mode().put(paddingModeNative)

  override private[torch] val nativeModule: Conv2dImpl = Conv2dImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"Conv2d($inChannels, $outChannels, kernelSize=$kernelSize, stride=$stride, padding=$padding, bias=$bias)"

object Conv2d:
  enum PaddingMode:
    case Zeros, Reflect, Replicate, Circular
