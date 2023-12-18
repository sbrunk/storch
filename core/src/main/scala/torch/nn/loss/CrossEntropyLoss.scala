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
package loss

import org.bytedeco.pytorch.CrossEntropyLossImpl
import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative

/** This criterion computes the cross entropy loss between input and target. */
// TODO optional args
final class CrossEntropyLoss extends Module {
  override private[torch] val nativeModule: CrossEntropyLossImpl = CrossEntropyLossImpl()

  override def hasBias(): Boolean = false

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )
}
