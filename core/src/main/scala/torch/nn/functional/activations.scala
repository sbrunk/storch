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
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.javacpp.LongPointer
import torch.internal.NativeConverters.toOptional
import org.bytedeco.pytorch.{ScalarTypeOptional, TensorOptional}

/** Applies a softmax followed by a logarithm.
  *
  * While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically
  * unstable. This function uses an alternative formulation to compute the output and gradient correctly.
  *
  * See `torch.nn.LogSoftmax` for more details.
  */
def logSoftmax[In <: DType, Out <: DType](input: Tensor[In], dim: Long)(dtype: Out = input.dtype): Tensor[Out] =
  val nativeDType = if dtype == input.dtype then ScalarTypeOptional() else ScalarTypeOptional(dtype.toScalarType)
  Tensor(torchNative.log_softmax(input.native, dim, nativeDType))

  /** Applies the rectified linear unit function element-wise.
    *
    * See `torch.nn.ReLU` for more details.
    */
def sigmoid[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.sigmoid(input.native))

/** Applies the element-wise function $\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$
  *
  * See `torch.nn.Sigmoid` for more details.
  */
def relu[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.relu(input.native))

/** Applies a softmax function. */
def softmax[In <: DType, Out <: DType](input: Tensor[In], dim: Long)(dtype: Out = input.dtype): Tensor[Out] =
  val nativeDType = if dtype == input.dtype then ScalarTypeOptional() else ScalarTypeOptional(dtype.toScalarType)
  Tensor(torchNative.softmax(input.native, dim, nativeDType))
