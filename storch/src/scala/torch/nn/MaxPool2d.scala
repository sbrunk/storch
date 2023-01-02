package torch
package nn

import org.bytedeco.pytorch.MaxPool2dImpl
import org.bytedeco.javacpp.LongPointer

import torch.internal.NativeConverters.toNative
import org.bytedeco.pytorch
import org.bytedeco.pytorch.MaxPool2dOptions

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final case class MaxPool2d[ParamType <: BFloat16 | Float32 | Float64](
    kernelSize: Int | (Int, Int),
    stride: Option[Int | (Int, Int)] = None,
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    //returnIndices: Boolean = false,
    ceilMode: Boolean = false
) extends HasParams[ParamType] with TensorModule[ParamType]:

  private val options: MaxPool2dOptions = MaxPool2dOptions(toNative(kernelSize))
  stride.foreach(s => options.stride().put(toNative(s)))
  options.padding().put(toNative(padding))
  options.dilation().put(toNative(dilation))
  options.ceil_mode().put(ceilMode)

  override private[torch] val nativeModule: MaxPool2dImpl = MaxPool2dImpl(options)
  nativeModule.asModule.to(paramType.toScalarType)

  override def registerWithParent[M <: pytorch.Module](parent: M)(using name: sourcecode.Name): Unit =
    parent.register_module(name.value, nativeModule)

  override def toString(): String = s"MaxPool2d(kernelSize=$kernelSize, stride=$stride, padding=$padding, dilation=$dilation, ceilMode=$ceilMode)"

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = Tensor(nativeModule.forward(t.native))
  // TODO forward_with_indices
