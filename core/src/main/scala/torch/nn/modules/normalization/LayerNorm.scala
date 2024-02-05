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

// cSpell:ignore elementwise_affine, nn, shapenormalized, storch, NLP

package torch
package nn
package modules
package normalization

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LayerNormImpl, LayerNormOptions, LongVector}
import internal.NativeConverters.fromNative

// format: off
/** Applies Layer Normalization over a mini-batch of inputs as described in the paper
  * [[Layer Normalization https://arxiv.org/abs/1607.06450]]
  *
  * TODO
  * $$
  * y=x−E[x]Var[x]+ϵ∗γ+β
  * y=Var[x]+ϵ
  * ​x−E[x]​∗γ+β
  * $$
  *
  * The mean and standard-deviation are calculated over the last D dimensions, where D is the
  * dimension of `normalized_shape`. For example, if `normalized_shape` is (3, 5) (a 2-dimensional
  * shape), the mean and standard-deviation are computed over the last 2 dimensions of the input
  * (i.e. input.mean((-2, -1))). γ and β are learnable affine transform parameters of
  * `normalized_shape` if `elementwise_affine` is `true`. The standard-deviation is calculated via
  * the biased estimator, equivalent to `torch.var(input, unbiased=False)`.
  *
  * @note
  *   Unlike Batch Normalization and Instance Normalization, which applies scalar scale and bias for
  *   each entire channel/plane with the `affine` option, Layer Normalization applies per-element
  *   scale and bias with `elementwise_affine`.
  *
  * @variable
  *   weight – the learnable weights of the module of shape `normalized_shape` when
  *   `elementwise_affine` is set to `true`. The values are initialized to 1. bias – the learnable
  *   bias of the module of shape `normalized_shape` when `elementwise_affine` is set to `true`. The
  *   values are initialized to 0.
  *
  * @example
  *   TODO
  *   ```scala
  *   // NLP Example
  *   val Seq(batch, sentence_length, embedding_dim) = Seq(20, 5, 10)
  *   val embedding = torch.randn(batch, sentence_length, embedding_dim)
  *   val layer_norm = nn.LayerNorm(embedding_dim)
  *   // Activate module
  *   val out = layer_norm(embedding)
  *
  *   // Image Example
  *   val Seq(N, C, H, W) = Seq(20, 5, 10, 10)
  *   val input = torch.randn(N, C, H, W)
  *   // Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
  *   val layer_norm = nn.LayerNorm([C, H, W])
  *   val output = layer_norm(input)
  *   ```
  *
  * @param `normalized_shape`
  *   – input shape from an expected input of size
  *   [∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]]
  *   [∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]] If a single integer is
  *   used, it is treated as a singleton list, and this module will normalize over the last
  *   dimension which is expected to be of that specific size.
  * @param eps
  *   – a value added to the denominator for numerical stability. Default: 1e-5
  * @param elementwise_affine
  *   – a boolean value that when set to `true`, this module has learnable per-element affine
  *   parameters initialized to ones (for weights) and zeros (for biases). Default: `true`.
  */
// format: on
final class LayerNorm[ParamType <: FloatNN | ComplexNN: Default](
    normalizedShape: Seq[Int],
    eps: Double = 1e-05,
    elementWiseAffine: Boolean = true
) extends HasWeight[ParamType]
    with TensorModule[ParamType]:

  private val shape: LongVector = LongVector(normalizedShape.map(_.toLong): _*)
  private val options: LayerNormOptions = LayerNormOptions(shape)
  options.eps().put(eps)
  options.elementwise_affine().put(elementWiseAffine)

  override private[torch] val nativeModule: LayerNormImpl = LayerNormImpl(options)

  val weight: Tensor[ParamType] = fromNative[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType] = fromNative[ParamType](nativeModule.bias)

  override def hasBias(): Boolean = true

  def apply(t: Tensor[ParamType]): Tensor[ParamType] =
    fromNative[ParamType](nativeModule.forward(t.native))
