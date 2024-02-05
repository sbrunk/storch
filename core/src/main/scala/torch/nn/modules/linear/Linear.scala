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
package linear

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LinearImpl, LinearOptions}
import internal.NativeConverters.fromNative

/** Applies a linear transformation to the incoming data: $y = xA^T + b$
  *
  * This module supports `TensorFloat32<tf32_on_ampere>`.
  *
  * Example:
  *
  * ```scala sc:nocompile
  * import torch.*
  *
  * val linear = nn.Linear[Float32](20, 30)
  * val input = torch.rand(Seq(128, 20))
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
final class Linear[ParamType <: FloatNN: Default](
    inFeatures: Long,
    outFeatures: Long,
    addBias: Boolean = true
    // dtype: ParamType = defaultDType[ParamType]
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:

  private val options = new LinearOptions(inFeatures, outFeatures)
  options.bias().put(addBias)

  override private[torch] val nativeModule: LinearImpl = new LinearImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  override def hasBias(): Boolean = options.bias().get()

  def weight = fromNative[ParamType](nativeModule.weight())
  def weight_=(t: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.weight(t.native)
    t

  def bias = fromNative[ParamType](nativeModule.bias())
  def bias_=(t: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.bias(t.native)
    t

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  override def toString =
    s"${getClass.getSimpleName}(inFeatures=$inFeatures, outFeatures=$outFeatures, bias=$addBias)"
