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

/** A simple lookup table that stores embeddings of a fixed dictionary and size.
  *
  * This module is often used to store word embeddings and retrieve them using indices. The input to
  * the module is a list of indices, and the output is the corresponding word embeddings.
  *
  * @group nn_sparse
  *
  * @param numEmbeddings
  *   Size of the dictionary of embeddings
  * @param embeddingDim
  *   The size of each embedding vector
  * @param paddingIdx
  *   If specified, the entries at `paddingIdx` do not contribute to the gradient; therefore, the
  *   embedding vector at `paddingIdx` is not updated during training, i.e. it remains as a fixed
  *   "pad". For a newly constructed Embedding, the embedding vector at `paddingIdx` will default to
  *   all zeros, but can be updated to another value to be used as the padding vector.
  * @param maxNorm
  *   If given, each embedding vector with norm larger than `maxNorm` is renormalized to have norm
  *   `maxNorm`.
  * @param normType
  *   The p of the p-norm to compute for the `maxNorm` option. Default `2`.
  * @param scaleGradByFreq
  *   If given, this will scale gradients by the inverse of frequency of the words in the
  *   mini-batch. Default `false`.
  * @param sparse
  *   If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor. See Notes for more
  *   details regarding sparse gradients.
  */
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
  nativeModule.to(paramType.toScalarType, false)

  def weight: Tensor[ParamType] = Tensor[ParamType](nativeModule.weight)
  def weight_=(w: Tensor[ParamType]): Unit = nativeModule.weight(w.native)

  def apply(t: Tensor[Int64]): Tensor[ParamType] = Tensor(nativeModule.forward(t.native))

  override def toString(): String = s"${getClass().getSimpleName()}(numEmbeddings=$numEmbeddings)"
