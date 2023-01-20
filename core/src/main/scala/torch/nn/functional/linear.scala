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
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.toOptional

// Linear functions

def linear[D <: DType](input: Tensor[D], weight: Tensor[D], bias: Tensor[D] | Option[Tensor[D]] = None): Tensor[D] =
  Tensor(
    torchNative.linear(input.native, weight.native, toOptional(bias))
  )

def bilinear[D <: DType](
    input1: Tensor[D],
    input2: Tensor[D],
    weight: Tensor[D],
    bias: Tensor[D] | Option[Tensor[D]] = None
): Tensor[D] = Tensor(
  torchNative.bilinear(input1.native, input2.native, weight.native, toOptional(bias))
)
