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
package functional

import org.bytedeco.pytorch
import org.bytedeco.pytorch.BCEWithLogitsLossOptions
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

// Loss functions
private[torch] trait Loss {

  /** Function that measures Binary Cross Entropy between target and input logits.
    *
    * TODO support weight, reduction, pos_weight
    *
    * @group nn_loss
    */
  def binaryCrossEntropyWithLogits[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] =
    fromNative(
      torchNative.binary_cross_entropy_with_logits(
        input.native,
        target.native,
        BCEWithLogitsLossOptions()
      )
    )

  // format: off
  // http://bytedeco.org/javacpp-presets/pytorch/apidocs/
  /** This criterion computes the cross entropy loss between input logits and target. See
    * [[torch.nn.loss.CrossEntropyLoss]] for details.
    *
    * **Shape:**
    *
    *   * Input: Shape $(C)$, $(N,C)$ or $(N,C,d_1,d_2,...,d_K)$ with $K≥1$ in the case of K-dimensional
    * loss.
    *   * Target: If containing class indices, shape $()$, $(N)$ or $(N,d_1,d_2,...,d_K)$ with $K≥1$ 
    * in the case of K-dimensional loss where each value should be between $[0,C)$. If containing class 
    * probabilities, same shape as the input and each value should be between [0,1][0,1].
    * 
    * where:
    *   * C = number of classes
    *   * N = batch size​
    *
    * @example
    *   ```scala
    *   // Example of target with class indices
    *   val input = torch.randn(3, 5, requires_grad=True)
    *   val target = torch.randint(5, (3,), dtype=torch.int64)
    *   val loss = F.cross_entropy(input, target)
    *   loss.backward()
    *
    *   // Example of target with class probabilities
    *   val input = torch.randn(3, 5, requires_grad=True)
    *   val target = torch.randn(3, 5).softmax(dim=1)
    *   val loss = F.crossEntropy(input, target)
    *   loss.backward()
    *   ```
    *
    * @param input
    *   Predicted unnormalized logits; see Shape section above for supported shapes.
    * @param target
    *   Ground truth class indices or class probabilities; see Shape section below for supported
    *   shapes.
    * @param weight
    *   a manual rescaling weight given to each class. If given, has to be a Tensor of size C
    * @param size_average
    *   Deprecated (see reduction). By default, the losses are averaged over each loss element in
    *   the batch. Note that for some losses, there multiple elements per sample. If the field
    *   `size_average` is set to `false`, the losses are instead summed for each mini-batch. Ignored
    *   when reduce is `false`. Default: `true`
    * @param ignore_index
    *   Specifies a target value that is ignored and does not contribute to the input gradient. When
    *   `size_average` is `true`, the loss is averaged over non-ignored targets. Note that
    *   `ignore_index` is only applicable when the target contains class indices. Default: `-100`
    * @param reduce
    *   Deprecated (see reduction). By default, the losses are averaged or summed over observations
    *   for each mini-batch depending on `size_average`. When reduce is `false`, returns a loss per
    *   batch element instead and ignores size_average. Default: `true`
    * @param reduction
    *   Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
    *   reduction will be applied, 'mean': the sum of the output will be divided by the number of
    *   elements in the output, 'sum': the output will be summed. Note: `size_average` and `reduce`
    *   are in the process of being deprecated, and in the meantime, specifying either of those two
    *   args will override reduction. Default: 'mean'
    * @param label_smoothing
    *   A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0
    *   means no smoothing. The targets become a mixture of the original ground truth and a uniform
    *   distribution as described in
    *   [[https://arxiv.org/abs/1512.00567 Rethinking the Inception Architecture for Computer Vision]].
    *   Default: 0.0
    *
    * @return
    *   [[torch.Tensor]]
    *
    * @see
    *   See
    *   [[https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html torch.nn.functional.cross_entropy]]
    * @see
    *   See
    *   [[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for equivalent torch.nn.CrossEntropyLoss class]]
    * @see
    *   See [[https://pytorch.org/cppdocs/ PyTorch C++ documentation]]
    * @see
    *   See [[http://bytedeco.org/javacpp-presets/pytorch/apidocs/ ByteDeco PyTorch preset]]
    */
    // format: on
  def crossEntropy[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[I] =
    fromNative(
      torchNative.cross_entropy(
        input.native,
        target.native
      )
    )

}
