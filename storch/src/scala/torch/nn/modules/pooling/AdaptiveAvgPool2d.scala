package torch
package nn
package modules
package pooling

import torch.nn.HasParams
import org.bytedeco.pytorch.AdaptiveAvgPool2dImpl
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch

import torch.internal.NativeConverters.{toNative, toOptional}
import org.bytedeco.pytorch.LongOptionalVector
import org.bytedeco.pytorch.LongOptional

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the number of input
  * planes.
  */
case class AdaptiveAvgPool2d(outputSize: Int | Option[Int] | (Option[Int], Option[Int]) | (Int, Int)) extends Module {

  private def nativeOutputSize = outputSize match
    case (h, w): (Int, Int) => new LongOptionalVector(new LongOptional(h), new LongOptional(w))
    case x: Int             => new LongOptionalVector(new LongOptional(x), new LongOptional(x))
    case (h, w): (Option[Int], Option[Int]) =>
      new LongOptionalVector(toOptional(h.map(_.toLong)), toOptional(w.map(_.toLong)))
    case x: Option[Int] => new LongOptionalVector(toOptional(x.map(_.toLong)), toOptional(x.map(_.toLong)))

  override private[torch] val nativeModule: AdaptiveAvgPool2dImpl = AdaptiveAvgPool2dImpl(nativeOutputSize.get(0))

  override def registerWithParent[T <: pytorch.Module](parent: T)(using name: sourcecode.Name): Unit =
    parent.register_module(name.value, nativeModule)

  def apply[D <: BFloat16 | Float32 | Float64](t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))
}
