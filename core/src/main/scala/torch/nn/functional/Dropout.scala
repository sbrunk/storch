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

import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

private[torch] trait Dropout {

  /** During training, randomly zeroes some of the elements of the input tensor with probability `p`
    * using samples from a Bernoulli distribution.
    *
    * @see
    *   [[torch.nn.Dropout]] for details.
    *
    * @group nn_dropout
    */
  def dropout[D <: DType](input: Tensor[D], p: Double = 0.5, training: Boolean = true): Tensor[D] =
    fromNative(
      torchNative.dropout(input.native, p, training)
    )

  // TODO alpha_dropout Applies alpha dropout to the input.
  // TODO feature_alpha_dropout Randomly masks out entire channels (a channel is a feature map, e.g.
  // TODO dropout1d Randomly zero out entire channels (a channel is a 1D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 1D tensor input[i,j]input[i,j]) of the input tensor).
  // TODO dropout2d Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 2D tensor input[i,j]input[i,j]) of the input tensor).
  // TODO dropout3d Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 3D tensor input[i,j]input[i,j]) of the input tensor).
}
