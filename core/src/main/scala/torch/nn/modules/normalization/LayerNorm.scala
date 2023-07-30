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
package normalization

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LayerNormImpl, LayerNormOptions, LongVector}
import torch.nn.modules.TensorModule
import torch.{DType, Tensor}

/** Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer
  * Normalization // TODO Add docs
  */
final class LayerNorm[ParamType <: DType: Default](
    normalizedShape: Seq[Int] | Int,
    eps: Double = 1e-05,
    elementwiseAffine: Boolean = true
) extends TensorModule[ParamType]:
  private val options: LayerNormOptions = normalizedShape match {
    case normalizedShape: Seq[Int] =>
      LayerNormOptions(LongVector(normalizedShape.map(_.toLong)*))
    case normalizedShape: Int =>
      LayerNormOptions(LongVector(normalizedShape.toLong))
  }
  options.eps().put(eps)
  options.elementwise_affine().put(elementwiseAffine)

  override private[torch] val nativeModule: LayerNormImpl = LayerNormImpl(options)

  val weight: Tensor[ParamType] = Tensor[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType] = Tensor[ParamType](nativeModule.bias)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] =
    Tensor[ParamType](nativeModule.forward(t.native))
