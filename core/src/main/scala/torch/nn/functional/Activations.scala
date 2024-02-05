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

import Derive.derive
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.ScalarTypeOptional

private[torch] trait Activations {

  /** Applies a softmax followed by a logarithm.
    *
    * While mathematically equivalent to log(softmax(x)), doing these two operations separately is
    * slower and numerically unstable. This function uses an alternative formulation to compute the
    * output and gradient correctly.
    *
    * See `torch.nn.LogSoftmax` for more details.
    *
    * @group nn_activation
    */
  def logSoftmax[In <: DType, Out <: FloatNN | Derive](
      input: Tensor[In],
      dim: Long,
      dtype: Out = derive
  ): Tensor[DTypeOrDeriveFromTensor[In, Out]] =
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    val nativeDType =
      if dtype == input.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(torchNative.log_softmax(input.native, dim, nativeDType))

    /** Applies the rectified linear unit function element-wise.
      *
      * See [[torch.nn.ReLU]] for more details.
      *
      * @group nn_activation
      */
  def relu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.relu(input.native))

  /** Applies the element-wise function $\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$
    *
    * See `torch.nn.Sigmoid` for more details.
    *
    * @group nn_activation
    */
  def sigmoid[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.sigmoid(input.native)
  )

  /** Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known
    * as the swish function.
    *
    * @group nn_activation
    */
  def silu[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.silu(input.native))

  /** Applies a softmax function.
    *
    * @group nn_activation
    */
  def softmax[In <: DType, Out <: FloatNN | Derive](
      input: Tensor[In],
      dim: Long,
      dtype: Out = derive
  ): Tensor[DTypeOrDeriveFromTensor[In, Out]] =
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    val nativeDType =
      if dtype == input.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(torchNative.softmax(input.native, dim, nativeDType))
}
