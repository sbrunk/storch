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

import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.FanModeType
import org.bytedeco.pytorch.kFanIn
import org.bytedeco.pytorch.kFanOut
import org.bytedeco.pytorch.Nonlinearity as NonlinearityNative
import org.bytedeco.pytorch.kLinear
import org.bytedeco.pytorch.kConv1D
import org.bytedeco.pytorch.kConv2D
import org.bytedeco.pytorch.kConv3D
import org.bytedeco.pytorch.kConvTranspose1D
import org.bytedeco.pytorch.kConvTranspose2D
import org.bytedeco.pytorch.kConvTranspose3D
import org.bytedeco.pytorch.kSigmoid
import org.bytedeco.pytorch.kReLU
import org.bytedeco.pytorch.kLeakyReLU
import org.bytedeco.pytorch.Scalar

// TODO implement remaining init functions
/** No gradients will be recorded for these operations.
  *
  * @see
  *   [[File init.h https://pytorch.org/cppdocs/api/file_torch_csrc_api_include_torch_nn_init.h.html#file-torch-csrc-api-include-torch-nn-init-h]]
  */
object init:

  // TODO: missing. How do we extract the tuple from the LongPointer?
  // @see https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ac77cdd61e014948ab8118e946c6d1d31.html#exhale-function-namespacetorch-1-1nn-1-1init-1ac77cdd61e014948ab8118e946c6d1d31

  // format: off
  /** Return the recommended gain value for the given nonlinearity function. The values are as
    * follows:
    *
    * nonlinearity                       gain
    * 
    * Linear / Identity                  1
    * Conv{1,2,3}D                       1
    * Sigmoid                            1
    * Tanh                               \frac{5}{3}
    * ReLU                               $sqrt{r}$
    * Leaky Relu                         $\sqrt{\frac{1}{1+\text{negative_slope}^2​}}$
    * SELU                               \frac{3}{4}
    * 
    * @note
    *   In order to implement
    *   [[Self-Normalizing Neural Networks https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html]],
    *   you should use `nonlinearity='linear'` instead of `nonlinearity='selu'`. This gives the
    *   initial weights a variance of 1/N, which is necessary to induce a stable fixed point in the
    *   forward pass. In contrast, the default gain for `SELU` sacrifices the normalization effect
    *   for more stable gradient flow in rectangular layers.
    *
    * @param nonlinearity
    *   – the non-linear function (nn.functional name)
    * @param param
    *   – optional parameter for the non-linear function
    *
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1abfccc1bd475b9a8c9505f46353e02c90.html#exhale-function-namespacetorch-1-1nn-1-1init-1abfccc1bd475b9a8c9505f46353e02c90
    */
  // format: on
  def calculateGain(
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU,
      param: Double = 0.01
  ): Double =
    torchNative.calculate_gain(nonlinearity.toNative, param)

  /** Fills the given 2-dimensional input Tensor with values drawn from the uniform distribution
    * $U(a,b)$. No gradient will be recorded for this operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param a
    *   – the lower bound of the uniform distribution
    * @param b
    *   – the upper bound of the uniform distribution
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ab0aeccca28b2225ee9aab809ec38a801.html#exhale-function-namespacetorch-1-1nn-1-1init-1ab0aeccca28b2225ee9aab809ec38a801
    */
  def uniform_[D <: DType](
      t: Tensor[D],
      a: Double = 0,
      b: Double = 1
  ): Tensor[D] =
    torchNative.uniform_(t.native, a, b)
    t

  /** Fills the he given 2-dimensional input Tensor with values drawn from the normal distribution
    * $N(\text{mean},\text{std}^2)$. No gradient will be recorded for this operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param mean
    *   – the mean of the normal distribution
    * @param std
    *   – the standard deviation of the normal distribution
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a105c2a8ef81c6faa82a01cf35ce9f3b1.html#exhale-function-namespacetorch-1-1nn-1-1init-1a105c2a8ef81c6faa82a01cf35ce9f3b1
    */
  def normal_[D <: DType](
      t: Tensor[D],
      mean: Double = 0,
      std: Double = 0
  ): t.type =
    torchNative.normal_(t.native, mean, std)
    t

  // TODO valid for all scala types
  /** Fills the input Tensor with the value valval.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param fillValue
    *   – the value to fill the tensor with
    *
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a9c886724aac3a487553dc0a406565c83.html#exhale-function-namespacetorch-1-1nn-1-1init-1a9c886724aac3a487553dc0a406565c83
    */
  def constant_[D <: DType](t: Tensor[D], fillValue: Double): t.type =
    torchNative.constant_(t.native, Scalar(fillValue))
    t

  /** Fills the input Tensor with the scalar value 1. No gradient will be recorded for this
    * operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a9dcc2051aadbe8ddb37d58bbd2b7943a.html#exhale-function-namespacetorch-1-1nn-1-1init-1a9dcc2051aadbe8ddb37d58bbd2b7943a
    */
  def ones_[D <: DType](
      t: Tensor[D]
  ): t.type =
    torchNative.ones_(t.native)
    t

  /** Fills the input Tensor with the scalar value 0. No gradient will be recorded for this
    * operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1af7e7736ba2d050adc0523d84285564e8.html#exhale-function-namespacetorch-1-1nn-1-1init-1af7e7736ba2d050adc0523d84285564e8
    */
  def zeros_[D <: DType](
      t: Tensor[D]
  ): t.type =
    torchNative.zeros_(t.native)
    t

  /** Fills the given 2-dimensional matrix with an identity matrix. No gradient will be recorded for
    * this operation.
    *
    * Fills the 2-dimensional input [[Tensor]] with the identity matrix. Preserves the identity of
    * the inputs in Linear layers, where as many inputs are preserved as possible.
    *
    * @param t
    *   – a 2-dimensional torch.Tensor
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a77eb9bba76a93da5b33e7770f9113015.html#exhale-function-namespacetorch-1-1nn-1-1init-1a77eb9bba76a93da5b33e7770f9113015
    */
  def eye_[D <: DType](
      t: Tensor[D]
  ): t.type =
    torchNative.eye_(t.native)
    t

  // TODO: no groups available
  /** From libTorch Fills the given tensor with the Dirac delta function in-place, and returns it.
    * No gradient will be recorded for this operation.
    *
    * From Pytorch
    *
    * Fills the {3, 4, 5}-dimensional input [[Tensor]] with the Dirac delta function. Preserves the
    * identity of the inputs in Convolutional layers, where as many input channels are preserved as
    * possible. In case of groups>1, each group of channels preserves identity
    *
    * @param t
    *   – a {3, 4, 5}-dimensional torch.Tensor
    * @param groups
    *   (int, optional) – number of groups in the conv layer (default: 1)
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ab9fa9ea51c05df8a5c9dcca7a54dd628.html#exhale-function-namespacetorch-1-1nn-1-1init-1ab9fa9ea51c05df8a5c9dcca7a54dd628
    */
  def dirac_[D <: DType](
      t: Tensor[D]
  ): t.type =
    torchNative.dirac_(t.native)
    t

  /** Fills the input [[Tensor]] with values according to the method described in "Understanding the
    * difficulty of training deep feedforward neural networks"" - Glorot, X. & Bengio, Y. (2010),
    * using a uniform distribution. Values are scaled by the gain parameter. The resulting tensor
    * will have values sampled from $U(−a,a)$ where $a=gain \times \sqrt{\frac{6}{fan_in+fan_out$}}
    *
    * Also known as Glorot initialization.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param gain
    *   – an optional scaling factor
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a86191a828a085e1c720dbce185d6c307.html#exhale-function-namespacetorch-1-1nn-1-1init-1a86191a828a085e1c720dbce185d6c307
    */
  def xavierNormal_[D <: DType](
      t: Tensor[D],
      gain: Double = 1.0
  ): t.type =
    torchNative.xavier_normal_(t.native, gain)
    t

  /** Fills the input [[Tensor]] with values according to the method described in "Understanding the
    * difficulty of training deep feedforward neural networks"" - Glorot, X. & Bengio, Y. (2010),
    * using a normal distribution. Values are scaled by the gain parameter. The resulting tensor
    * will have values sampled from $N(0,\text{std}^2) $ where $a=gain \times
    * \sqrt{\frac{2}{fan_in+fan_out$}}
    *
    * Also known as Glorot initialization.
    *
    * @param t
    * @param gain
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html#exhale-function-namespacetorch-1-1nn-1-1init-1ace282f75916a862c9678343dfd4d5ffe
    */
  def xavierUniform_[D <: DType](
      t: Tensor[D],
      gain: Double = 1.0
  ): t.type =
    torchNative.xavier_uniform_(t.native, gain)
    t

  /** Fills the input Tensor with values according to the method described in Delving deep into
    * rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al.
    * (2015), using a uniform distribution. The resulting tensor will have values sampled from
    * $U(−bound,bound)$ where $\text{bound} = \text{gain} \times \sqrt{\frac{3}{fan_mode}}
    *
    * Also known as He initialization.
    *
    * No gradient will be recorded for this operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param a
    *   – the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
    * @param mode
    *   – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
    *   variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in
    *   the backwards pass.
    * @param nonlinearity
    *   – the non-linear function (nn.functional name), recommended to use only with 'relu' or
    *   'leaky_relu' (default).
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a5e807af188fc8542c487d50d81cb1aa1.html#exhale-function-namespacetorch-1-1nn-1-1init-1a5e807af188fc8542c487d50d81cb1aa1
    */
  def kaimingUniform_[D <: DType](
      t: Tensor[D],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): t.type =
    torchNative.kaiming_uniform_(t.native, a, mode.toNative, nonlinearity.toNative)
    t

  /** Fills the input Tensor with values according to the method described in "Delving deep into
    * rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al.
    * (2015), using a normal distribution. The resulting tensor will have values sampled from
    * $N(0,std^22)$ where: $$std = \frac{gain}{\sqrt{fan_mode}}
    *
    * Also known as He initialization.
    *
    * No gradient will be recorded for this operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param a
    *   – the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
    * @param mode
    *   – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
    *   variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in
    *   the backwards pass.
    * @param nonlinearity
    *   – the non-linear function (nn.functional name), recommended to use only with 'relu' or
    *   'leaky_relu' (default).
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ac8a913c051976a3f41f20df7d6126e57.html#exhale-function-namespacetorch-1-1nn-1-1init-1ac8a913c051976a3f41f20df7d6126e57
    */
  def kaimingNormal_[D <: DType](
      t: Tensor[D],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): t.type =
    torchNative.kaiming_normal_(t.native, a, mode.toNative, nonlinearity.toNative)
    t

  // TODO: no trunc normal as per the PyTorch API. C++ docs not commented. Not part of init but of function
  // /**
  //   * Fills the input Tensor with values drawn from a truncated normal distribution. The values are
  //   * effectively drawn from the normal distribution $N(\text{mean},\text{std}^2)$ with values outside
  //   * $[a,b]$ redrawn until they are within the bounds. The method used for generating the random values
  //   * works best when $a \le \text{mean} \le \text{b}.
  //   *
  //   * @param t – an n-dimensional torch.Tensor
  //   * @param mean (float) – the mean of the normal distribution
  //   * @param std (float) – the standard deviation of the normal distribution
  //   * @param a (float) – the minimum cutoff value
  //   * @param b (float) – the maximum cutoff value
  //   * @see https://pytorch.org/cppdocs/api/function_namespaceat_1aa604fcef7ea09fc379dc92c5d92a06ab.html
  //   */
  def trunc_[D <: DType](
      t: Tensor[D]
  ): t.type =
    torchNative.trunc_(t.native)
    t

  /** Fills the input Tensor with a (semi) orthogonal matrix, as described in Exact solutions to the
    * nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013). The
    * input tensor must have at least 2 dimensions, and for tensors with more than 2 dimensions the
    * trailing dimensions are flattened. No gradient will be recorded for this operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor, where n≥2n≥2
    * @param gain
    *   – optional scaling factor
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a5978fcc257460475f635b5960e892a8e.html#exhale-function-namespacetorch-1-1nn-1-1init-1a5978fcc257460475f635b5960e892a8e
    */
  def orthogonal_[D <: DType](
      t: Tensor[D],
      gain: Double = 1.0
  ): t.type =
    torchNative.orthogonal_(t.native, gain)
    t

  /** Fills the 2D input Tensor as a sparse matrix, where the non-zero elements will be drawn from
    * the normal distribution $N(0,0.01)$, as described in "Deep learning via Hessian-free
    * optimization" - Martens, J. (2010). The sparsity is a real value between 0 and 1 that controls
    * the fraction of elements in each column to be set to zero.
    *
    * No gradient will be recorded for this operation.
    *
    * @param t
    *   – an n-dimensional torch.Tensor
    * @param gain
    *   – The fraction of elements in each column to be set to zero
    * @param std
    *   – the standard deviation of the normal distribution used to generate the non-zero values
    * @see
    *   https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a82f2e5810880c7cc60c84516eb283be6.html#exhale-function-namespacetorch-1-1nn-1-1init-1a82f2e5810880c7cc60c84516eb283be6
    */
  def sparse_[D <: DType](
      t: Tensor[D],
      sparsity: Double,
      std: Double = 0.01
  ): t.type =
    torchNative.sparse_(t.native, sparsity, std)
    t

  enum Mode:
    case FanIn, FanOut
    private[torch] def toNative: FanModeType = FanModeType(this match
      case Mode.FanIn  => kFanIn()
      case Mode.FanOut => kFanOut()
    )

  enum NonLinearity:
    case Linear, Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D, Sigmoid,
      ReLU, LeakyReLU
    private[torch] def toNative: NonlinearityNative = NonlinearityNative(this match
      case NonLinearity.Linear          => kLinear()
      case NonLinearity.Conv1D          => kConv1D()
      case NonLinearity.Conv2D          => kConv2D()
      case NonLinearity.Conv3D          => kConv3D()
      case NonLinearity.ConvTranspose1D => kConvTranspose1D()
      case NonLinearity.ConvTranspose2D => kConvTranspose2D()
      case NonLinearity.ConvTranspose3D => kConvTranspose3D()
      case NonLinearity.Sigmoid         => kSigmoid()
      case NonLinearity.ReLU            => kReLU()
      case NonLinearity.LeakyReLU       => kLeakyReLU()
    )
