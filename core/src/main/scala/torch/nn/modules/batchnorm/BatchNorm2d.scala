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
package batchnorm

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import sourcecode.Name
import org.bytedeco.pytorch.BatchNorm2dImpl
import org.bytedeco.pytorch.BatchNormOptions
import torch.nn.modules.{HasParams, HasWeight, TensorModule}


// format: off
/** Applies Batch Normalization over a 2D or 3D input as described in the paper
[Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/abs/1502.03167) .

$$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$

The mean and standard-deviation are calculated per-dimension over
the mini-batches and $\gamma$ and $\beta$ are learnable parameter vectors
of size [C]{.title-ref} (where [C]{.title-ref} is the number of features or channels of the input). By default, the
elements of $\gamma$ are set to 1 and the elements of $\beta$ are set to 0. The
standard-deviation is calculated via the biased estimator, equivalent to [torch.var(input, unbiased=False)]{.title-ref}.

Also by default, during training this layer keeps running estimates of its
computed mean and variance, which are then used for normalization during
evaluation. The running estimates are kept with a default `momentum`{.interpreted-text role="attr"}
of 0.1.

If `track_running_stats`{.interpreted-text role="attr"} is set to `False`, this layer then does not
keep running estimates, and batch statistics are instead used during
evaluation time as well.

::: note
::: title
Note
:::

This `momentum`{.interpreted-text role="attr"} argument is different from one used in optimizer
classes and the conventional notion of momentum. Mathematically, the
update rule for running statistics here is
$\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t$,
where $\hat{x}$ is the estimated statistic and $x_t$ is the
new observed value.
:::

Because the Batch Normalization is done over the [C]{.title-ref} dimension, computing statistics
on [(N, L)]{.title-ref} slices, it\'s common terminology to call this Temporal Batch Normalization.

Args:

:   num_features: number of features or channels $C$ of the input
    eps: a value added to the denominator for numerical stability.
    Default: 1e-5
    momentum: the value used for the running_mean and running_var
    computation. Can be set to `None` for cumulative moving average
    (i.e. simple average). Default: 0.1
    affine: a boolean value that when set to `True`, this module has
    learnable affine parameters. Default: `True`
    track_running_stats: a boolean value that when set to `True`, this
    module tracks the running mean and variance, and when set to `False`,
    this module does not track such statistics, and initializes statistics
    buffers `running_mean`{.interpreted-text role="attr"} and `running_var`{.interpreted-text role="attr"} as `None`.
    When these buffers are `None`, this module always uses batch statistics.
    in both training and eval modes. Default: `True`

Shape:

:   -   Input: $(N, C)$ or $(N, C, L)$, where $N$ is the batch size,
        $C$ is the number of features or channels, and $L$ is the sequence length
    -   Output: $(N, C)$ or $(N, C, L)$ (same shape as input)

Examples:

    >>> # With Learnable Parameters
    >>> m = nn.BatchNorm1d(100)
    >>> # Without Learnable Parameters
    >>> m = nn.BatchNorm1d(100, affine=False)
    >>> input = torch.randn(20, 100)
    >>> output = m(input)
  *
  * @group nn_conv
  * 
  * TODO use dtype
  */
// format: on
final class BatchNorm2d[ParamType <: FloatNN | ComplexNN: Default](
    numFeatures: Int,
    eps: Double = 1e-05,
    momentum: Double = 0.1,
    affine: Boolean = true,
    trackRunningStats: Boolean = true
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:

  private val options = new BatchNormOptions(numFeatures)
  options.eps().put(eps)
  options.momentum().put(momentum)
  options.affine().put(affine)
  options.track_running_stats().put(trackRunningStats)

  override private[torch] val nativeModule: BatchNorm2dImpl = BatchNorm2dImpl(options)
  nativeModule.asModule.to(paramType.toScalarType)

  override def registerWithParent[M <: pytorch.Module](parent: M)(using
      name: sourcecode.Name
  ): Unit =
    parent.register_module(name.value, nativeModule)

  // TODO weight, bias etc. are undefined if affine = false. We need to take that into account
  val weight: Tensor[ParamType] = Tensor[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType] = Tensor[ParamType](nativeModule.bias)
  // TODO running_mean, running_var, num_batches_tracked

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = Tensor(nativeModule.forward(t.native))

  override def toString(): String = s"${getClass().getSimpleName()}(numFeatures=$numFeatures)"
