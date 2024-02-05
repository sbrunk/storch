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

import org.bytedeco.javacpp.{LongPointer, DoublePointer}
import org.bytedeco.pytorch.GenericDict
import spire.math.Complex
import spire.math.UByte
import scala.annotation.targetName
import org.bytedeco.pytorch.GeneratorOptional
import org.bytedeco.pytorch.TensorArrayRef
import org.bytedeco.pytorch.TensorVector

private[torch] object NativeConverters:

  inline def convertToOptional[T, U <: T | Option[T], V >: Null](i: U, f: T => V): V = i match
    case i: Option[T] => i.map(f(_)).orNull
    case i: T         => f(i)

  extension (l: Int | Option[Int])
    def toOptional: pytorch.LongOptional = convertToOptional(l, pytorch.LongOptional(_))

  extension (l: Double | Option[Double])
    def toOptional: pytorch.DoubleOptional = convertToOptional(l, pytorch.DoubleOptional(_))

  extension (l: Real | Option[Real])
    def toOptional: pytorch.ScalarOptional =
      convertToOptional(
        l,
        (r: Real) =>
          val scalar = toScalar(r)
          pytorch.ScalarOptional(scalar)
      )

  extension (i: Int) def toScalarOptional = pytorch.ScalarOptional(pytorch.Scalar(i))

  extension [D <: DType](t: Tensor[D] | Option[Tensor[D]])
    def toOptional: TensorOptional =
      convertToOptional(t, t => pytorch.TensorOptional(t.native))

  extension (i: Int | Seq[Int] | (Int, Int) | (Int, Int, Int))
    @targetName("intOrIntSeqToArray")
    def toArray: Array[Long] = i match
      case i: Int      => Array(i.toLong)
      case i: Seq[Int] => i.map(_.toLong).toArray
      case (i, j)      => Array(i.toLong, j.toLong)
      case (i, j, k)   => Array(i, j, k).map(_.toLong)

  extension (input: Int | (Int, Int) | (Int, Int, Int))
    def toNative = input match
      case x: Int    => LongPointer(Array(x.toLong, x.toLong)*)
      case (h, w)    => LongPointer(Array(h.toLong, w.toLong)*)
      case (t, h, w) => LongPointer(Array(t.toLong, h.toLong, w.toLong)*)

  given doubleToDoublePointer: Conversion[Double, DoublePointer] = (input: Double) =>
    DoublePointer(Array(input)*)

  extension (x: ScalaType)
    def toScalar: pytorch.Scalar = x match
      case x: Boolean => pytorch.AbstractTensor.create(x).item()
      case x: UByte   => pytorch.AbstractTensor.create(x.toInt).to(uint8.toScalarType).item()
      case x: Byte    => pytorch.Scalar(x)
      case x: Short   => pytorch.Scalar(x)
      case x: Int     => pytorch.Scalar(x)
      case x: Long    => pytorch.Scalar(x)
      case x: Float   => pytorch.Scalar(x)
      case x: Double  => pytorch.Scalar(x)
      case x @ Complex(r: Float, i: Float)   => Tensor(Seq(x)).to(dtype = complex64).native.item()
      case x @ Complex(r: Double, i: Double) => Tensor(Seq(x)).to(dtype = complex128).native.item()

  extension (g: Option[Generator] | Generator)
    def toOptional = g match
      case g: Generator => new pytorch.GeneratorOptional(g.native)
      case None         => new pytorch.GeneratorOptional()
      case Some(g)      => new pytorch.GeneratorOptional(g.native)

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

  def toArrayRef(tensors: Seq[Tensor[?]]): TensorArrayRef =
    new TensorArrayRef(TensorVector(tensors.map(_.native)*))

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

  export Tensor.fromNative
