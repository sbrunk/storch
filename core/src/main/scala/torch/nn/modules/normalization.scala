package torch
package nn
package modules

import org.bytedeco.pytorch.GroupNormOptions
import org.bytedeco.pytorch.GroupNormImpl
import org.bytedeco.pytorch

/** Applies Group Normalization over a mini-batch of inputs
  *
  * @param numGroups
  *   number of groups to separate the channels into
  * @param numChannels
  *   number of channels expected in input
  * @param eps
  *   a value added to the denominator for numerical stability
  * @param affine
  *   a boolean value that when set to `true`, this module has learnable per-channel affine parameters initialized to
  *   ones (for weights) and zeros (for biases)
  */
class GroupNorm[ParamType <: DType](numGroups: Int, numChannels: Int, eps: Double = 1e-05, affine: Boolean = true)
    extends TensorModule[ParamType]:
  private val options: GroupNormOptions = GroupNormOptions(numGroups, numChannels)
  options.eps().put(eps)
  options.affine().put(affine)

  override private[torch] val nativeModule: GroupNormImpl = GroupNormImpl(options)

  override def registerWithParent[M <: pytorch.Module](parent: M)(using name: sourcecode.Name): Unit =
    parent.register_module(name.value, nativeModule)

  val weight: Tensor[ParamType] = Tensor[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType]   = Tensor[ParamType](nativeModule.bias)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = Tensor[ParamType](nativeModule.forward(t.native))
