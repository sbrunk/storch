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
package linear

import org.bytedeco.pytorch
import org.bytedeco.pytorch.IdentityImpl
import torch.internal.NativeConverters.fromNative
import scala.annotation.nowarn

/** A placeholder identity operator that is argument-insensitive.
  *
  * @group nn_linear
  */
final class Identity[D <: DType: Default](@nowarn("msg=unused explicit parameter") args: Any*)
    extends TensorModule[D]:
  override val nativeModule: IdentityImpl = IdentityImpl()

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
