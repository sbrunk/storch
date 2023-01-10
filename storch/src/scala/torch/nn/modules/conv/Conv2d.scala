package torch
package nn
package modules
package conv

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.*
import sourcecode.Name
import torch.Tensor
import torch.internal.NativeConverters.toNative
import torch.nn.modules.conv.Conv2d.PaddingMode
import torch.nn.modules.{HasParams, TensorModule}

/** Applies a 2D convolution over an input signal composed of several input planes.
  *
  * @group nn_conv
  * 
  */
case class Conv2d[ParamType <: FloatNN | ComplexNN : Default](
    inChannels: Long,
    outChannels: Long,
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int) = 1,
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    groups: Int = 1,
    bias: Boolean = true,
    paddingMode: PaddingMode = PaddingMode.Zeros
) extends HasParams[ParamType] with TensorModule[ParamType]:

  private val options = new Conv2dOptions(inChannels, outChannels, toNative(kernelSize))
  options.stride().put(toNative(stride))
  options.padding().put(toNative(padding))
  options.dilation().put(toNative(dilation))
  options.groups().put(groups)
  options.bias().put(bias)
  private val paddingModeNative = paddingMode match
    case PaddingMode.Zeros => new kZeros
    case PaddingMode.Reflect => new kReflect
    case PaddingMode.Replicate => new kReplicate
    case PaddingMode.Circular => new kCircular
  options.padding_mode().put(paddingModeNative)

  override private[torch] val nativeModule: Conv2dImpl = Conv2dImpl(options)
  nativeModule.asModule.to(paramType.toScalarType)

  override def registerWithParent[M <: pytorch.Module](parent: M)(using name: sourcecode.Name): Unit =
    // println(s"registering ${name.value}: $this with $parent")
    parent.register_module(name.value, nativeModule)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = Tensor(nativeModule.forward(t.native))

  val weight = Tensor[ParamType](nativeModule.weight)

  override def toString = s"Conv2d($inChannels, $outChannels, kernelSize=$kernelSize, stride=$stride, padding=$padding, bias=$bias)"

object Conv2d:
  enum PaddingMode:
    case Zeros, Reflect, Replicate, Circular