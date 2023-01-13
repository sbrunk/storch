package torch
package nn
package modules
package linear

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LinearImpl, LinearOptions}
import torch.Tensor
import torch.nn.modules.{HasParams, TensorModule}

/** Applies a linear transformation to the incoming data: $y = xA^T + b$
  *
  * This module supports `TensorFloat32<tf32_on_ampere>`.
  *
  * * Example:
  *
  * ```scala sc:nocompile
  * import torch.*
  *
  * val linear = nn.Linear[Float32](20, 30)
  * val input  = torch.rand(Seq(128, 20))
  * println(linear(input).size) // ArraySeq(128, 30)
  * ```
  *
  * @group nn_linear
  *
  * @param inFeatures
  *   size of each input sample
  * @param outFeatures
  *   size of each output sample
  * @param bias
  *   If set to ``false``, the layer will not learn an additive bias. Default: ``true``
  */
class Linear[ParamType <: FloatNN: Default](
    inFeatures: Long,
    outFeatures: Long,
    bias: Boolean = true,
    //dtype: ParamType = defaultDType[ParamType]
) extends HasParams[ParamType] with (TensorModule[ParamType]):
  private val options = new LinearOptions(inFeatures, outFeatures)
  options.bias().put(bias)
  override private[torch] val nativeModule: LinearImpl = new LinearImpl(options)
  nativeModule.asModule.to(paramType.toScalarType)

  override def registerWithParent[T <: pytorch.Module](parent: T)(using name: sourcecode.Name): Unit =
    parent.register_module(name.value, nativeModule)

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = Tensor(nativeModule.forward(input.native))

  override def toString = s"${getClass.getSimpleName}(inFeatures=$inFeatures, outFeatures=$outFeatures, bias=$bias)"
