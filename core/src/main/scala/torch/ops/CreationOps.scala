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

import internal.NativeConverters
import NativeConverters.*
import Layout.Strided
import Device.CPU
import MemoryFormat.Contiguous

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{IValue, GenericDict, BoolOptional, Scalar, MemoryFormatOptional}
import org.bytedeco.pytorch.global.torch as torchNative

import java.nio.file.{Files, Path}
import scala.collection.immutable.{VectorMap, SeqMap}
import Tensor.fromNative

/** Creation Ops
  *
  * https://pytorch.org/docs/stable/torch.html#creation-ops
  */
private[torch] trait CreationOps {

// TODO sparse_coo_tensor
// TODO as_tensor
// TODO as_strided
// TODO frombuffer

// def zeros[D <: DType](size: Int*): Tensor[Float32] =
//   zeros[D](size.toSeq)

  /** Returns a tensor filled with the scalar value `0`, with the shape defined by the variable
    * argument `size`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @tparam T
    * @return
    *
    * @group creation_ops
    */
  def zeros[D <: DType](
      size: Seq[Int] | Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
      case s: Int      => Array(s.toLong)
    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

  /** @group creation_ops
    */
  def zerosLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requiresGrad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requiresGrad, memoryFormat, torchNative.torch_zeros_like)

  /** Returns a tensor filled with the scalar value `1`, with the shape defined by the variable
    * argument `size`.
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    *
    * @group creation_ops
    */
  def ones[D <: DType](
      size: Seq[Int] | Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
      case s: Int      => Array(s.toLong)
    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

  /** @group creation_ops */
  def onesLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requiresGrad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requiresGrad, memoryFormat, torchNative.torch_ones_like)

// format: off
/** Returns a 1-D tensor of size $`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`$ with values
  * from the interval ``[start, end)`` taken with common difference `step` beginning from `start`.
  *
  * Note that non-integer `step` is subject to floating point rounding errors when comparing against `end`;
  * to avoid inconsistency, we advise adding a small epsilon to `end` in such cases.
  *
  * $$
  * \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
  * $$
  *
  * @param start
  *   The starting value for the set of points. Default: ``0``.
  * @param end
  *   The ending value for the set of points
  * @param step
  *   The gap between each pair of adjacent points. Default: ``1``.
  *   
  * @group creation_ops
  */
// format: on
  def arange[D <: DType | Derive, Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
      start: Start = 0,
      end: End,
      step: Step = 1,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[DTypeOrDeriveArange[D, Start, End, Step]] =
    val derivedDType = dtype match
      case _: Derive => derivedArangeType(start, end, step)
      case t: DType  => t
    fromNative(
      torchNative.torch_arange(
        toScalar(start),
        toScalar(end),
        toScalar(step),
        NativeConverters.tensorOptions(derivedDType, layout, device, requiresGrad)
      )
    )

  /** @group creation_ops */
  def linspace[D <: DType](
      start: Double,
      end: Double,
      steps: Long,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_linspace(
        new Scalar(start),
        new Scalar(end),
        steps,
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

  /** @group creation_ops */
  def logspace[D <: DType](
      start: Double,
      end: Float,
      steps: Long,
      base: Double = 10.0,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ) = fromNative(
    torchNative.torch_logspace(
      new Scalar(start),
      new Scalar(end),
      steps,
      base,
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
    )
  )

  /** Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    *
    * @param n
    *   the number of rows
    * @param m
    *   the number of columns with default being `n`
    * @param dtype
    *   the desired data type of the returned tensor.
    * @param layout
    *   the desired layout of the returned tensor.
    * @param device
    *   the desired device of the returned tensor.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    *
    * @group creation_ops
    */
  def eye[D <: DType](
      n: Int,
      m: Option[Int] = None,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[D] = fromNative(
    torchNative.torch_eye(n, NativeConverters.tensorOptions(dtype, layout, device, requiresGrad))
  )
// def empty(size: Long*): Tensor[D] = fromNative(torchNative.torch_empty(size*))

  /** Returns a tensor filled with uninitialized data.
    *
    * @group creation_ops
    */
  def empty[D <: DType](
      size: Seq[Int],
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false,
      pinMemory: Boolean = false,
      memoryFormat: MemoryFormat = Contiguous
  ): Tensor[D] =
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(dtype, layout, device, requiresGrad)
          .pinned_memory(BoolOptional(pinMemory)),
        new MemoryFormatOptional(memoryFormat.toNative)
      )
    )

  /** Returns an uninitialized tensor with the same size as input.
    *
    * `torch.empty_like(input)` is equivalent to `torch.empty(input.size(), dtype=input.dtype,
    * layout=input.layout, device=input.device`).
    *
    * @group creation_ops
    */
  def emptyLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requiresGrad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requiresGrad, memoryFormat, torchNative.torch_empty_like)

// // TODO emptyStrided

  /** Creates a tensor of size `size` filled with `fillValue`. The tensor's dtype is inferred from
    * `fillValue`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @param fillValue
    *   the value to fill the output tensor with.
    * @param dtype
    *   the desired data type of the returned tensor.
    * @param layout
    *   the desired layout of the returned Tensor.
    * @param device
    *   the desired device of the returned tensor.
    * @param requiresGrad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the data type of the returned tensor, or `Default` if the type should be derived from
    *   `fillValue`.
    * @tparam U
    *   the data type of `fillValue`.
    * @return
    *   the newly created tensor.
    *
    * @group creation_ops
    */
  def full[D <: DType | Derive, U <: ScalaType](
      size: Seq[Int],
      fillValue: U,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =
    val derivedDType = dtype match
      case _: Derive => scalaToDType(fillValue)
      case t: DType  => t
    fromNative(
      torchNative.torch_full(
        size.toArray.map(_.toLong),
        toScalar(fillValue),
        NativeConverters.tensorOptions(derivedDType, layout, device, requiresGrad)
      )
    )

// TODO fullLike
// TODO quantize_per_tensor
// TODO quantize_per_channel
// TODO dequantize
// TODO complex
// TODO polar
// TODO heavside

  def pickleLoad(data: Array[Byte]): SeqMap[String, Tensor[DType]] =
    val dict: GenericDict = torchNative.pickle_load(data).toGenericDict()
    // We need to extract the members in one go or we risk too early deallocation of native objects here
    val buffer = new Array[(IValue, IValue)](dict.size().toInt)
    val nativeIt = dict.begin()
    for (i <- 0 until buffer.size)
      buffer(i) = (nativeIt.access().key(), nativeIt.access().value())
      nativeIt.increment()
    VectorMap.from(buffer.map { (key, value) =>
      // TODO better error handling
      (key.toStringRef().getString(), fromNative[DType](value.toTensor().clone()))
    })

  def pickleLoad(path: Path): Map[String, Tensor[DType]] =
    val data: Array[Byte] = Files.readAllBytes(path)
    pickleLoad(data)

  def pickleSave(tensors: SeqMap[String, Tensor[DType]]) =
    tensors.map { (k, v) =>
      (IValue(k), IValue(v.native))
    }

}
