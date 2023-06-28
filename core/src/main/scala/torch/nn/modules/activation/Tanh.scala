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
package activation

import org.bytedeco.pytorch
import org.bytedeco.pytorch.TanhImpl
import torch.nn.modules.Module
import torch.{DType, Tensor}

/** Applies the Hyperbolic Tangent (Tanh) function element-wise. Tanh is defined as::
  *
  * TODO LaTeX
  */
final class Tanh[D <: DType: Default]() extends TensorModule[D]:

  override protected[torch] val nativeModule: TanhImpl = new TanhImpl()

  override def registerWithParent[M <: pytorch.Module](parent: M)(using
      name: sourcecode.Name
  ): Unit =
    parent.register_module(name.value, nativeModule)

  def apply(t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))

  override def toString = getClass().getSimpleName()