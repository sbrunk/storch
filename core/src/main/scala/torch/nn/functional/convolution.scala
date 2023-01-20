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
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

/** Applies a 1D convolution over an input signal composed of several input planes. */
def conv1d[D <: FloatNN | ComplexNN](
    input: Tensor[D],
    weight: Tensor[D],
    bias: Tensor[D] | Option[Tensor[D]] = None,
    stride: Long = 1,
    padding: Long = 0,
    dilation: Long = 1,
    groups: Long = 1
): Tensor[D] =
  Tensor(
    torchNative.conv1d(
      input.native,
      weight.native,
      toOptional(bias),
      Array(stride),
      Array(padding),
      Array(dilation),
      groups
    )
  )

/** Applies a 2D convolution over an input signal composed of several input planes. */
def conv2d[D <: FloatNN | ComplexNN](
    input: Tensor[D],
    weight: Tensor[D],
    bias: Tensor[D] | Option[Tensor[D]] = None,
    stride: Long | (Long, Long) = 1,
    padding: Long | (Long, Long) = 0,
    dilation: Long | (Long, Long) = 1,
    groups: Long = 1
): Tensor[D] =
  Tensor(
    torchNative.conv2d(
      input.native,
      weight.native,
      toOptional(bias),
      toArray(stride),
      toArray(padding),
      toArray(dilation),
      groups
    )
  )
