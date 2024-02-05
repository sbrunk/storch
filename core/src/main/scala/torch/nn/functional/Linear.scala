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
import torch.internal.NativeConverters.{fromNative, toOptional}

// Linear functions
private[torch] trait Linear {

  /** Applies a linear transformation to the incoming data: $y = xA^T + b$.
    *
    * This operation supports 2-D `weight` with `sparse layout`
    *
    * Warning
    *
    * Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be
    * supported, or may not have autograd support. If you notice missing functionality please open a
    * feature request.
    *
    * This operator supports `TensorFloat32<tf32_on_ampere>`
    *
    * Shape:
    *
    *   - Input: $(*, in\_features)$ where [\*] means any number of additional dimensions, including
    *     none
    *   - Weight: $(out\_features, in\_features)$ or $(in\_features)$
    *   - Bias: $(out\_features)$ or $()$
    *   - Output: $(*, out\_features)$ or $(*)$, based on the shape of the weight
    *
    * @group nn_linear
    */
  def linear[D <: DType](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None
  ): Tensor[D] =
    fromNative(
      torchNative.linear(input.native, weight.native, toOptional(bias))
    )

  /** Applies a bilinear transformation to the incoming data: $y = x_1^T A x_2 + b$
    *
    * Shape:
    *
    *   - input1: $(N, *, H_{in1})$ where $H_{in1}=\text{in1\_features}$ and $*$ means any number of
    *     additional dimensions. All but the last dimension of the inputs should be the same.
    *   - input2: $(N, *, H_{in2})$ where $H_{in2}=\text{in2\_features}$
    *   - weight: $(\text{out\_features}, \text{in1\_features}, \text{in2\_features})$
    *   - bias: $(\text{out\_features})$
    *   - output: $(N, *, H_{out})$ where $H_{out}=\text{out\_features}$ and all but the last
    *     dimension are the same shape as the input.
    *
    * @group nn_linear
    */
  def bilinear[D <: DType](
      input1: Tensor[D],
      input2: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D] | Option[Tensor[D]] = None
  ): Tensor[D] = fromNative(
    torchNative.bilinear(input1.native, input2.native, weight.native, toOptional(bias))
  )

}
