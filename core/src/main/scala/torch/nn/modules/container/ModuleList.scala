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
package container

import sourcecode.Name
import scala.util.Random

final class ModuleList[D <: DType](override val modules: TensorModule[D]*)
    extends Module
    with Seq[TensorModule[D]]:
  // with TensorModule[D]:
  modules.zipWithIndex.foreach((module, index) =>
    this.register(module)(using Name(index.toString()))
  )

  override def iterator: Iterator[TensorModule[D]] = modules.iterator

  // def apply(v1: Int | torch.Tensor[D]): torch.nn.modules.TensorModule[D] & torch.Tensor[D] = ???
  def apply(i: Int): torch.nn.modules.TensorModule[D] = modules(i)

  override def length: Int = modules.length

  override def toString = getClass().getSimpleName()
