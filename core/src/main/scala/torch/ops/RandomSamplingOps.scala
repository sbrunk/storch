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
package ops

import Layout.Strided
import Device.CPU
import internal.NativeConverters
import NativeConverters.*

import org.bytedeco.pytorch.global.torch as torchNative

/** Random Sampling
  *
  * https://pytorch.org/docs/stable/torch.html#random-sampling
  */
private[torch] trait RandomSamplingOps {

// TODO seed Sets the seed for generating random numbers to a non-deterministic random number.
// TODO manual_seed Sets the seed for generating random numbers.
// TODO initial_seed Returns the initial seed for generating random numbers as a Python long.
// TODO get_rng_state Returns the random number generator state as a torch.ByteTensor.
// TODO set_rng_state Sets the random number generator state.
// TODO bernoulli Draws binary random numbers (0 or 1) from a Bernoulli distribution.

  /* Returns a tensor where each row contains `numSamples` indices sampled from the multinomial probability distribution located in the corresponding row of tensor `input`. */
  def multinomial[D <: FloatNN](
      input: Tensor[D],
      numSamples: Long,
      replacement: Boolean = false,
      generator: Option[Generator] | Generator = None
  ): Tensor[Int64] =
    fromNative(torchNative.multinomial(input.native, numSamples, replacement, generator.toOptional))

// TODO normal Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
// TODO poisson Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input i.e.,

  /** Returns a tensor filled with random numbers from a uniform distribution on the interval
    * `[0,1)`
    *
    * The shape of the tensor is defined by the variable argument `size`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @param dtype
    *   the desired data type of returned tensor.
    * @param layout
    *   the desired layout of returned Tensor.
    * @param device
    *   the desired device of returned tensor.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the dtype of the created tensor.
    */
  def rand[D <: FloatNN | ComplexNN](
      size: Seq[Int],
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_rand(
        size.toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

  /** Returns a tensor with the same size as `input` that is filled with random numbers from a
    * uniform distribution on the interval $[0, 1)$.
    *
    * `torch.randLike(input)` is equivalent to `torch.rand(input.size(), dtype=input.dtype,
    * layout=input.layout, device=input.device)`.
    *
    * @param input
    *   the size of `input` will determine size of the output tensor.
    * @param dtype
    *   the desired data type of returned Tensor. If `derive`, defaults to the dtype of `input`.
    * @param layout
    *   the desired layout of returned tensor. If `derive`, defaults to the layout of `input`.
    * @param device
    *   the desired device of returned tensor. If `derive` , defaults to the device of `input`.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @param memoryFormat
    *   the desired memory format of returned Tensor.
    */
  def randLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requiresGrad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requiresGrad, memoryFormat, torchNative.torch_rand_like)

  /** Returns a tensor filled with random integers generated uniformly between `low` (inclusive) and
    * `high` (exclusive).
    *
    * The shape of the tensor is defined by the variable argument `size`.
    *
    * @param low
    *   Lowest integer to be drawn from the distribution. Default: 0.
    * @param high
    *   One above the highest integer to be drawn from the distribution.
    * @param size
    *   a tuple defining the shape of the output tensor.
    * @param generator
    *   a pseudorandom number generator for sampling
    * @param dtype
    *   the desired data type of returned tensor.
    * @param layout
    *   the desired layout of returned Tensor.
    * @param device
    *   the desired device of returned tensor.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the dtype of the created tensor.
    */
  def randint[D <: DType](
      low: Long = 0,
      high: Long,
      size: Seq[Int],
      generator: Option[Generator] | Generator = None,
      dtype: D = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randint(
        low,
        high,
        size.toArray.map(_.toLong),
        generator.toOptional,
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

// TODO randint_like Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive).

// TODO Randnd acepts Seq[Int] | Int
  def randn[D <: FloatNN | ComplexNN](
      size: Seq[Int],
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randn(
        size.toArray.map(_.toLong),
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

// TODO randn_like Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.

  /** Returns a random permutation of integers from 0 to n - 1.
    *
    * TODO support custom generator
    */
  def randperm[D <: DType](
      n: Long,
      dtype: D = int64,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false,
      pinMemory: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_randperm(
        n,
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad, pinMemory)
      )
    )

  def manualSeed(seed: Long) = torchNative.manual_seed(seed)

  def setNumThreads(threads: Int): Unit = torchNative.set_num_threads(threads)
}
