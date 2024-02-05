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
package functional

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  AvgPool1dOptions,
  AvgPool2dOptions,
  AvgPool3dOptions,
  MaxPool1dOptions,
  MaxPool2dOptions,
  MaxPool3dOptions
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

/** @define maxPoolPadding
  *   Implicit negative infinity padding to be added on both sides, must be \>= 0 and \<=
  *   kernel_size / 2.
  * @define dilation
  *   The stride between elements within a sliding window, must be \> 0.
  * @define ceilMode
  *   If `true`, will use *ceil* instead of *floor* to compute the output shape. This ensures that
  *   every element in the input tensor is covered by a sliding window.
  */
private[torch] trait Pooling {

  /** Applies a 1D max pooling over an input signal composed of several input planes.
    *
    * @see
    *   [[torch.nn.AdaptiveAvgPool1d]] for details and output shape.
    *
    * @param input
    *   input tensor of shape $(\text{minibatch} , \text{in\_channels} , iW)$
    * @param kernelSize
    *   the size of the window.
    * @param stride
    *   the stride of the window. Default: `kernelSize`
    * @param padding
    *   implicit zero paddings on both sides of the input. Can be a single number or a tuple
    *   `(padW,)`.
    * @param ceilMode
    *   $ceilMode
    * @param countIncludePad
    *   when true, will include the zero-padding in the averaging calculation.
    *
    * @group nn_pooling
    */
  def avgPool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int),
      stride: Int | None.type = None,
      padding: Int = 0,
      ceilMode: Boolean = false,
      countIncludePad: Boolean = true
  ): Tensor[D] =
    val options = AvgPool1dOptions(toNative(kernelSize))
    stride match
      case s: Int => options.stride().put(toNative(s))
      case None   =>
    options.padding().put(toNative(padding))
    options.ceil_mode().put(ceilMode)
    options.count_include_pad().put(countIncludePad)
    fromNative(torchNative.avg_pool1d(input.native, options))

  /** Applies a 2D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def avgPool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type = None,
      padding: Int | (Int, Int) = 0,
      ceilMode: Boolean = false,
      countIncludePad: Boolean = true,
      divisor_override: Int | None.type = None
  ): Tensor[D] =
    val options = AvgPool2dOptions(toNative(kernelSize))
    stride match
      case s: (Int | (Int, Int)) => options.stride().put(toNative(s))
      case None                  =>
    options.padding().put(toNative(padding))
    options.ceil_mode().put(ceilMode)
    options.count_include_pad().put(countIncludePad)
    divisor_override match
      case d: Int => options.divisor_override().put(d)
      case None   =>
    fromNative(torchNative.avg_pool2d(input.native, options))

  /** Applies a 3D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def avgPool3d[D <: Float16 | Float32 | Float64 | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      ceilMode: Boolean = false,
      countIncludePad: Boolean = true,
      divisor_override: Int | None.type = None
  ): Tensor[D] =
    val options = AvgPool3dOptions(toNative(kernelSize))
    stride match
      case s: (Int | (Int, Int, Int)) => options.stride().put(toNative(s))
      case None                       =>
    options.padding().put(toNative(padding))
    options.ceil_mode().put(ceilMode)
    options.count_include_pad().put(countIncludePad)
    divisor_override match
      case d: Int => options.divisor_override().put(d)
      case None   =>
    fromNative(torchNative.avg_pool3d(input.native, options))

  private def maxPool1dOptions[D <: FloatNN | Complex32](
      kernelSize: Int | (Int, Int),
      stride: Int | None.type,
      padding: Int,
      dilation: Int,
      ceilMode: Boolean
  ): MaxPool1dOptions =
    val options: MaxPool1dOptions = MaxPool1dOptions(toNative(kernelSize))
    stride match
      case s: Int => options.stride().put(toNative(s))
      case None   =>
    options.padding().put(toNative(padding))
    options.dilation().put(toNative(dilation))
    options.ceil_mode().put(ceilMode)
    options

  /** Applies a 1D max pooling over an input signal composed of several input planes.
    *
    * @param input
    *   input tensor of shape $(\text{minibatch} , \text{in\_channels} , iW)$, minibatch dim
    *   optional.
    * @param kernelSize
    *   the size of the window.
    * @param stride
    *   the stride of the window.
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    *
    * @group nn_pooling
    */
  def maxPool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int),
      stride: Int | None.type = None,
      padding: Int = 0,
      dilation: Int = 1,
      ceilMode: Boolean = false
  ): Tensor[D] =
    val options: MaxPool1dOptions =
      maxPool1dOptions(kernelSize, stride, padding, dilation, ceilMode)
    fromNative(torchNative.max_pool1d(input.native, options))

  /** Applies a 1D max pooling over an input signal composed of several input planes.
    *
    * @param input
    *   input tensor of shape $(\text{minibatch} , \text{in\_channels} , iW)$, minibatch dim
    *   optional.
    * @param kernelSize
    *   the size of the window.
    * @param stride
    *   the stride of the window.
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    *
    * @group nn_pooling
    */
  def maxPool1dWithIndices[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int),
      stride: Int | None.type = None,
      padding: Int = 0,
      dilation: Int = 1,
      ceilMode: Boolean = false
  ): TensorTuple[D] =
    val options: MaxPool1dOptions =
      maxPool1dOptions(kernelSize, stride, padding, dilation, ceilMode)
    val native = torchNative.max_pool1d_with_indices(input.native, options)
    TensorTuple(values = fromNative[D](native.get0()), indices = fromNative(native.get1))

  private def maxPool2dOptions[D <: FloatNN | Complex32](
      kernelSize: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type,
      padding: Int | (Int, Int),
      dilation: Int | (Int, Int),
      ceilMode: Boolean
  ): MaxPool2dOptions = {
    val options: MaxPool2dOptions = MaxPool2dOptions(toNative(kernelSize))
    stride match
      case s: (Int | (Int, Int)) => options.stride().put(toNative(s))
      case None                  =>
    options.padding().put(toNative(padding))
    options.dilation().put(toNative(dilation))
    options.ceil_mode().put(ceilMode)
    options
  }

  /** Applies a 2D max pooling over an input signal composed of several input planes.
    *
    * @see
    *   [[torch.nn.MaxPool2d]] for details.
    *
    * @param input
    *   input tensor $(\text{minibatch} , \text{in\_channels} , iH , iW)$, minibatch dim optional.
    * @param kernelSize
    *   size of the pooling region. Can be a single number or a tuple `(kH, kW)`
    * @param stride
    *   stride of the pooling operation. Can be a single number or a tuple `(sH, sW)`
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    *
    * @group nn_pooling
    */
  def maxPool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type = None,
      padding: Int | (Int, Int) = 0,
      dilation: Int | (Int, Int) = 1,
      ceilMode: Boolean = false
  ): Tensor[D] =
    val options: MaxPool2dOptions =
      maxPool2dOptions(kernelSize, stride, padding, dilation, ceilMode)
    fromNative(torchNative.max_pool2d(input.native, options))

  /** Applies a 2D max pooling over an input signal composed of several input planes.
    *
    * @see
    *   [[torch.nn.MaxPool2d]] for details.
    *
    * @param input
    *   input tensor $(\text{minibatch} , \text{in\_channels} , iH , iW)$, minibatch dim optional.
    * @param kernelSize
    *   size of the pooling region. Can be a single number or a tuple `(kH, kW)`
    * @param stride
    *   stride of the pooling operation. Can be a single number or a tuple `(sH, sW)`
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    *
    * @group nn_pooling
    */
  def maxPool2dWithIndices[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type = None,
      padding: Int | (Int, Int) = 0,
      dilation: Int | (Int, Int) = 1,
      ceilMode: Boolean = false
  ): TensorTuple[D] =
    val options: MaxPool2dOptions =
      maxPool2dOptions(kernelSize, stride, padding, dilation, ceilMode)
    val native = torchNative.max_pool2d_with_indices(input.native, options)
    TensorTuple(values = fromNative[D](native.get0()), indices = fromNative(native.get1))

  private def maxPool3dOptions[D <: FloatNN | Complex32](
      kernelSize: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type,
      padding: Int | (Int, Int, Int),
      dilation: Int | (Int, Int, Int),
      ceilMode: Boolean
  ): MaxPool3dOptions = {
    val options: MaxPool3dOptions = MaxPool3dOptions(toNative(kernelSize))
    stride match
      case s: (Int | (Int, Int, Int)) => options.stride().put(toNative(s))
      case None                       =>
    options.padding().put(toNative(padding))
    options.dilation().put(toNative(dilation))
    options.ceil_mode().put(ceilMode)
    options
  }

  /** Applies a 3D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def maxPool3d[D <: Float32 | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      dilation: Int | (Int, Int, Int) = 1,
      ceilMode: Boolean = false
  ): Tensor[D] =
    val options: MaxPool3dOptions =
      maxPool3dOptions(kernelSize, stride, padding, dilation, ceilMode)
    fromNative(torchNative.max_pool3d(input.native, options))

  /** Applies a 3D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def maxPool3dWithIndices[D <: Float16 | Float32 | Float64 | Complex32](
      input: Tensor[D],
      kernelSize: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      dilation: Int | (Int, Int, Int) = 1,
      ceilMode: Boolean = false
  ): TensorTuple[D] =
    val options: MaxPool3dOptions =
      maxPool3dOptions(kernelSize, stride, padding, dilation, ceilMode)
    val native = torchNative.max_pool3d_with_indices(input.native, options)
    TensorTuple(values = fromNative[D](native.get0()), indices = fromNative(native.get1))

  // TODO max_unpool1d Computes a partial inverse of MaxPool1d.
  // TODO max_unpool2d Computes a partial inverse of MaxPool2d.
  // TODO max_unpool3d Computes a partial inverse of MaxPool3d.
  // TODO lp_pool1d Applies a 1D power-average pooling over an input signal composed of several input planes.
  // TODO lp_pool2d Applies a 2D power-average pooling over an input signal composed of several input planes.
  // TODO adaptive_max_pool1d Applies a 1D adaptive max pooling over an input signal composed of several input planes.
  // TODO adaptive_max_pool2d Applies a 2D adaptive max pooling over an input signal composed of several input planes.
  // TODO adaptive_max_pool3d Applies a 3D adaptive max pooling over an input signal composed of several input planes.
  // TODO adaptive_avg_pool1d Applies a 1D adaptive average pooling over an input signal composed of several input planes.
  // TODO adaptive_avg_pool2d Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  // TODO adaptive_avg_pool3d Applies a 3D adaptive average pooling over an input signal composed of several input planes.
  // TODO fractional_max_pool2d Applies 2D fractional max pooling over an input signal composed of several input planes.
  // TODO fractional_max_pool3d Applies 3D fractional max pooling over an input signal composed of several input planes.

}
