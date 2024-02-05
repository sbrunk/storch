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

// cSpell:ignore nn, inplace

package torch
package nn
package modules
package regularization

import org.bytedeco.pytorch
import org.bytedeco.pytorch.DropoutImpl
import org.bytedeco.pytorch.DropoutOptions
import torch.internal.NativeConverters.fromNative

// format: off
/** During training, randomly zeroes some of the elements of the input tensor with probability `p` 
  * using samples from a Bernoulli distribution. Each channel will be zeroed out independently on 
  * every forward call.
  * 
  * This has proven to be an effective technique for regularization and preventing the co-adaptation 
  * of neurons as described in the paper [[https://arxiv.org/abs/1207.0580 Improving neural networks 
  * by preventing co-adaptation of feature detectors]].
  * 
  * Furthermore, the outputs are scaled by a factor of $\frac{1}{1−p}​ during training. This means 
  * that during evaluation the module simply computes an identity function.
  * 
  * Shape:
  * - Input: $(∗)(∗)$. Input can be of any shape
  * - Output: $(∗)(∗)$. Output is of the same shape as input
  *   
  * @example
  *
  * ```scala
  * import torch.nn
  * 
  * val m = nn.Dropout(p=0.2)
  * val input = torch.randn(20, 16)
  * val output = m(input)
  * ```
  * 
  * @param p – probability of an element to be zeroed. Default: 0.5
  * @param inplace – If set to True, will do this operation in-place. Default: `false`
  *  
  * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
  * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#dropout Pytorch Python Dropout]]
  * @see See [[https://pytorch.org/docs/master/nn.html#torch.nn.Dropout]]
  * @see See [[https://pytorch.org/docs/master/nn.html#torch.nn.Dropout2d]]
  * @see See [[https://pytorch.org/docs/master/nn.html#torch.nn.Dropout3d]]
  * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html#torch-nn-functional-dropout]]
  * 
  * TODO: https://pytorch.org/docs/master/nn.html#torch.nn.Dropout
  * Add 2D, 3D, Alpha and feature alpha versions
  */
// format: on
final class Dropout[ParamType <: FloatNN | ComplexNN: Default](
    p: Double = 0.5,
    inplace: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType]:

  private val options: DropoutOptions = DropoutOptions(p)
  options.inplace().put(inplace)

  override private[torch] val nativeModule: DropoutImpl = DropoutImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  override def hasBias(): Boolean = false

  override def toString(): String = s"${getClass().getSimpleName()}(p=$p, inplace=$inplace)"
