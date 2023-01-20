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

package torch.nn.modules.activation

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ReLUImpl, ReLUOptions}
import torch.nn.modules.Module
import torch.{DType, Tensor}

/** Applies the rectified linear unit function element-wise:
  *
  * $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final class ReLU(inplace: Boolean = false) extends Module:
  private val options = new ReLUOptions()
  options.inplace().put(inplace)

  override protected[torch] val nativeModule: ReLUImpl = ReLUImpl()

  override def registerWithParent[M <: pytorch.Module](parent: M)(using name: sourcecode.Name): Unit =
    // println(s"registering ${name.value}: $this with $parent")
    parent.register_module(name.value, nativeModule)

  def apply[D <: DType](t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))
