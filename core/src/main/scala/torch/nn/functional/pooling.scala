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

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{MaxPool1dOptions, MaxPool2dOptions, MaxPool3dOptions, TensorOptional}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

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

def maxPool1d[D <: FloatNN | Complex32](
    input: Tensor[D],
    kernelSize: Int | (Int, Int),
    stride: Int | None.type = None,
    padding: Int = 0,
    dilation: Int = 1,
    ceilMode: Boolean = false
): Tensor[D] =
  val options: MaxPool1dOptions = maxPool1dOptions(kernelSize, stride, padding, dilation, ceilMode)
  Tensor(torchNative.max_pool1d(input.native, options))

def maxPool1dWithIndices[D <: FloatNN | Complex32](
    input: Tensor[D],
    kernelSize: Int | (Int, Int),
    stride: Int | None.type = None,
    padding: Int = 0,
    dilation: Int = 1,
    ceilMode: Boolean = false
): TensorTuple[D] =
  val options: MaxPool1dOptions = maxPool1dOptions(kernelSize, stride, padding, dilation, ceilMode)
  val native = torchNative.max_pool1d_with_indices(input.native, options)
  TensorTuple(values = Tensor[D](native.get0()), indices = Tensor(native.get1))

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
def maxPool2d[D <: FloatNN | Complex32](
    input: Tensor[D],
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int) | None.type = None,
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    ceilMode: Boolean = false
): Tensor[D] =
  val options: MaxPool2dOptions = maxPool2dOptions(kernelSize, stride, padding, dilation, ceilMode)
  Tensor(torchNative.max_pool2d(input.native, options))

def maxPool2dWithIndices[D <: FloatNN | Complex32](
    input: Tensor[D],
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int) | None.type = None,
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    ceilMode: Boolean = false
): TensorTuple[D] =
  val options: MaxPool2dOptions = maxPool2dOptions(kernelSize, stride, padding, dilation, ceilMode)
  val native = torchNative.max_pool2d_with_indices(input.native, options)
  TensorTuple(values = Tensor[D](native.get0()), indices = Tensor(native.get1))

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
    case None                       => //options.stride().put(toNative(kernelSize))
  options.padding().put(toNative(padding))
  options.dilation().put(toNative(dilation))
  options.ceil_mode().put(ceilMode)
  options
}
def maxPool3d[D <: Float32 | Complex32](
    input: Tensor[D],
    kernelSize: Int | (Int, Int, Int),
    stride: Int | (Int, Int, Int) | None.type = None,
    padding: Int | (Int, Int, Int) = 0,
    dilation: Int | (Int, Int, Int) = 1,
    ceilMode: Boolean = false
): Tensor[D] =
  val options: MaxPool3dOptions = maxPool3dOptions(kernelSize, stride, padding, dilation, ceilMode)
  Tensor(torchNative.max_pool3d(input.native, options))

def maxPool3dWithIndices[D <: Float16 | Float32 | Float64 | Complex32](
    input: Tensor[D],
    kernelSize: Int | (Int, Int, Int),
    stride: Int | (Int, Int, Int) | None.type = None,
    padding: Int | (Int, Int, Int) = 0,
    dilation: Int | (Int, Int, Int) = 1,
    ceilMode: Boolean = false
): TensorTuple[D] =
  val options: MaxPool3dOptions = maxPool3dOptions(kernelSize, stride, padding, dilation, ceilMode)
  val native = torchNative.max_pool3d_with_indices(input.native, options)
  TensorTuple(values = Tensor[D](native.get0()), indices = Tensor(native.get1))
