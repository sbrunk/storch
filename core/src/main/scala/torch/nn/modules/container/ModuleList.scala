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

/** Holds submodules in a list.
  *
  * It can be indexed like a regular Python list, but the modules it contains are properly
  * registered, and will be visible by all [[torch.nn.Module]] methods.
  *
  * @example
  *   ```scala
  *   class MyModule extends nn.Module:
  *     val linears = register( nn.ModuleList([nn.Linear(10, 10) for i in range(10)]) )
  *
  *     // ModuleList can act as an iterable, or be indexed using ints
  *     def forward(self, x) =
  *       var x_ = x.copy_(x)
  *       for l <- linears
  *         x_ = x_ + l(x_)
  *       x
  *   ```
  *
  * @see
  *   [[torch.nn.ModuleList https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html?highlight=modulelist#torch.nn.ModuleList]]
  * @see
  *   [[container ModuleList https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#ModuleList]]
  */
final class ModuleList[D <: DType](override val modules: TensorModule[D]*)
    extends Module
    // with TensorModule[D]:
    // TODO
    with TensorModule[D]
    with scala.collection.immutable.Iterable[TensorModule[D]]:

  modules.zipWithIndex.foreach((module, index) =>
    this.register(module)(using Name(index.toString()))
  )

  override def iterator: Iterator[TensorModule[D]] = modules.iterator

  override def apply(input: Tensor[D]): Tensor[D] =
    modules.foldLeft(input)((i, module) => module(i))

  override def toString = getClass().getSimpleName()

  /** Insert a given module before a given index in the list.
    *
    * @param index
    *   index to insert.
    * @param module
    *   module to insert
    * @return
    *   ModuleList[D] with new elements
    */
  def insert(index: Int, module: TensorModule[D]): ModuleList[D] =
    val (before, after) = modules.splitAt(index)
    val all = before ++ (after.prepended(module))
    // TODO: not in Python code. Note other modules retain index, so we have repeats
    this.register(module)(using Name(index.toString()))
    // TODO: make modules list mutable?
    ModuleList(all: _*)

  /** Appends a given module to the end of the list.
    *
    * @param module
    *   module to append
    * @return
    *   ModuleList[D] with new elements
    */
  def append(module: TensorModule[D]): ModuleList[D] =
    // TODO: not in Module
    // self.add_module(str(len(self)), module)
    // TODO: not in Python code
    val index = modules.length
    this.register(module)(using Name(index.toString()))
    val all = modules.appended(module)
    // TODO: make modules list mutable?
    ModuleList(all: _*)

  /** Appends modules from a Python iterable to the end of the list.
    *
    * @param modules
    *   iterable of modules to append
    * @return
    */
  def extend(newModules: Iterable[TensorModule[D]]): ModuleList[D] =
    // TODO: not in Module
    // offset = len(self)
    // for i, module in enumerate(modules):
    //     self.add_module(str(offset + i), module)
    // return self
    // val offset = modules.length
    val all = modules ++ newModules
    // Not in Python
    newModules.zipWithIndex.foreach((module, index) =>
      this.register(module)(using Name(index.toString()))
    )
    // TODO: make modules list mutable?
    ModuleList(all: _*)

  override def hasBias(): Boolean = modules.exists(_.hasBias())

  def apply(i: Int): torch.nn.modules.TensorModule[D] = modules(i)

  def length: Int = modules.length
