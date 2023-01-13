package torch
package nn
package modules
package container

import sourcecode.Name
import scala.util.Random

case class Sequential[D <: DType](override val modules: TensorModule[D]*) extends Module with TensorModule[D]:
  modules.zipWithIndex.foreach((module, index) => this.register(module)(using Name(index.toString())))

  override def apply(input: Tensor[D]): Tensor[D] = modules.foldLeft(input)((i, module) => module(i))

  override def toString = getClass().getSimpleName()
