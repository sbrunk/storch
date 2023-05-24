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
package internal

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ScalarTypeOptional,
  LayoutOptional,
  DeviceOptional,
  DoubleOptional,
  BoolOptional,
  LongOptional,
  TensorOptional
}

import scala.reflect.Typeable
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch.GenericDict
import org.bytedeco.pytorch.GenericDictIterator
import spire.math.Complex
import spire.math.UByte

private[torch] object NativeConverters:

  inline def toOptional[T, U <: T | Option[T], V >: Null](i: U, f: T => V): V = i match
    case i: Option[T] => i.map(f(_)).orNull
    case i: T         => f(i)

  def toOptional(l: Long | Option[Long]): LongOptional = toOptional(l, pytorch.LongOptional(_))
  def toOptional(l: Double | Option[Double]): DoubleOptional =
    toOptional(l, pytorch.DoubleOptional(_))

  def toOptional[D <: DType](t: Tensor[D] | Option[Tensor[D]]): TensorOptional =
    toOptional(t, t => pytorch.TensorOptional(t.native))

  def toArray(i: Long | (Long, Long)) = i match
    case i: Long              => Array(i)
    case (i, j): (Long, Long) => Array(i, j)

  def toNative(input: Int | (Int, Int)) = input match
    case (h, w): (Int, Int) => LongPointer(Array(h.toLong, w.toLong)*)
    case x: Int             => LongPointer(Array(x.toLong, x.toLong)*)

  def toScalar(x: ScalaType): pytorch.Scalar = x match
    case x: Boolean                        => pytorch.Scalar(if true then 1: Byte else 0: Byte)
    case x: UByte                          => Tensor(x.toInt).to(dtype = uint8).native.item()
    case x: Byte                           => pytorch.Scalar(x)
    case x: Short                          => pytorch.Scalar(x)
    case x: Int                            => pytorch.Scalar(x)
    case x: Long                           => pytorch.Scalar(x)
    case x: Float                          => pytorch.Scalar(x)
    case x: Double                         => pytorch.Scalar(x)
    case x @ Complex(r: Float, i: Float)   => ???
    case x @ Complex(r: Double, i: Double) => ???

  def tensorOptions(
      dtype: DType,
      layout: Layout,
      device: Device,
      requiresGrad: Boolean,
      pinMemory: Boolean = false
  ): pytorch.TensorOptions =
    pytorch
      .TensorOptions()
      .dtype(ScalarTypeOptional(dtype.toScalarType))
      .layout(LayoutOptional(layout.toNative))
      .device(DeviceOptional(device.toNative))
      .requires_grad(BoolOptional(requiresGrad))
      .pinned_memory(BoolOptional(pinMemory))

  def tensorOptions(
      dtype: Option[DType],
      layout: Option[Layout],
      device: Option[Device],
      requiresGrad: Boolean
  ): pytorch.TensorOptions =
    pytorch
      .TensorOptions()
      .dtype(dtype.fold(ScalarTypeOptional())(d => ScalarTypeOptional(d.toScalarType)))
      .layout(layout.fold(LayoutOptional())(l => LayoutOptional(l.toNative)))
      .device(device.fold(DeviceOptional())(d => DeviceOptional(d.toNative)))
      .requires_grad(BoolOptional(requiresGrad))

  class NativeIterable[Container, NativeIterator, Item](
      container: Container,
      containerSize: Container => Long,
      begin: Container => NativeIterator,
      increment: NativeIterator => NativeIterator,
      access: NativeIterator => Item
  ) extends scala.collection.Iterable[Item]:

    override def iterator: Iterator[Item] = new Iterator[Item] {
      val it = begin(container)
      val len = containerSize(container)
      var index = 0

      override def next(): Item =
        val item = access(it)
        index += 1
        increment(it)
        item

      override def hasNext: Boolean = index < len
    }

  class GenericDictIterable(d: GenericDict)
      extends NativeIterable(
        container = d,
        containerSize = d => d.size(),
        begin = d => d.begin(),
        increment = it => it.increment(),
        access = it => it.access()
      )
