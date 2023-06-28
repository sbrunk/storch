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
package sparse

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import sourcecode.Name
import org.bytedeco.pytorch.EmbeddingImpl
import org.bytedeco.pytorch.EmbeddingOptions
import torch.nn.modules.{HasParams, HasWeight, TensorModule}
import torch.internal.NativeConverters.{toNative, doubleToDoublePointer}

final class Embedding[ParamType <: FloatNN | ComplexNN: Default](
    numEmbeddings: Int,
    embeddingDim: Int,
    paddingIdx: Option[Int] = None,
    maxNorm: Option[Double] = None,
    normType: Option[Double] = Some(2.0),
    scaleGradByFreq: Boolean = false,
    sparse: Boolean = false
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModuleBase[Int64, ParamType]:

  private val options = new EmbeddingOptions(numEmbeddings.toLong, embeddingDim.toLong)
  paddingIdx.foreach(p => options.padding_idx().put(toNative(p)))
  maxNorm.foreach(m => options.max_norm().put(m))
  normType.foreach(n => options.norm_type().put(n))
  options.scale_grad_by_freq().put(scaleGradByFreq)
  options.sparse().put(sparse)

  override val nativeModule: EmbeddingImpl = EmbeddingImpl(options)
  nativeModule.asModule.to(paramType.toScalarType)

  override def registerWithParent[M <: pytorch.Module](parent: M)(using
      name: sourcecode.Name
  ): Unit =
    parent.register_module(name.value, nativeModule)

  val weight: Tensor[ParamType] = Tensor[ParamType](nativeModule.weight)

  def apply(t: Tensor[Int64]): Tensor[ParamType] = Tensor(nativeModule.forward(t.native))

  override def toString(): String = s"${getClass().getSimpleName()}(numEmbeddings=$numEmbeddings)"
