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

import org.bytedeco.javacpp.*
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.global.torch.{ScalarType, toComplexType}
import org.bytedeco.pytorch.{
  BoolOptional,
  DeviceOptional,
  LayoutOptional,
  LinearImpl,
  LogSoftmaxFuncOptions,
  LongOptional,
  MemoryFormatOptional,
  Module,
  Scalar,
  ScalarTypeOptional,
  TensorArrayRef,
  TensorVector
}

import java.nio.{
  ByteBuffer,
  CharBuffer,
  DoubleBuffer,
  FloatBuffer,
  IntBuffer,
  LongBuffer,
  ShortBuffer
}
import scala.annotation.{targetName, varargs}
import scala.reflect.ClassTag
import internal.NativeConverters.*
import Layout.Strided
import Device.CPU
import torch.internal.NativeConverters
import MemoryFormat.Contiguous

import java.nio.file.Path
import java.nio.file.Files
import org.bytedeco.pytorch.GenericDict
import org.bytedeco.pytorch.IValue

import scala.collection.immutable.VectorMap
import scala.collection.immutable.SeqMap
import scala.util.Using

// Creation Ops

// // TODO sparse_coo_tensor
// // TODO as_tensor
// // TODO as_strided
// // TODO frombuffer

/** Returns a tensor filled with the scalar value `0`, with the shape defined by the variable
  * argument `size`.
  * @param size
  *   a sequence of integers defining the shape of the output tensor.
  * @tparam T
  * @return
  */
// def zeros[D <: DType](size: Int*): Tensor[Float32] =
//   zeros[D](size.toSeq)
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
  Tensor(
    torchNative.torch_zeros(
      nativeSize,
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
    )
  )

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
  * @tparam T
  * @return
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
  Tensor(
    torchNative.torch_ones(
      nativeSize,
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
    )
  )

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
  * from the interval ``[start, end)`` taken with common difference :attr:`step` beginning from `start`.
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
  Tensor(
    torchNative.torch_arange(
      toScalar(start),
      toScalar(end),
      toScalar(step),
      NativeConverters.tensorOptions(derivedDType, layout, device, requiresGrad)
    )
  )

def linspace[D <: DType](
    start: Double,
    end: Double,
    steps: Long,
    dtype: D = float32,
    layout: Layout = Strided,
    device: Device = CPU,
    requiresGrad: Boolean = false
): Tensor[D] =
  Tensor(
    torchNative.torch_linspace(
      new Scalar(start),
      new Scalar(end),
      steps,
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
    )
  )

def logspace[D <: DType](
    start: Double,
    end: Float,
    steps: Long,
    base: Double = 10.0,
    dtype: D = float32,
    layout: Layout = Strided,
    device: Device = CPU,
    requiresGrad: Boolean = false
) = Tensor(
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
  */
def eye[D <: DType](
    n: Int,
    m: Option[Int] = None,
    dtype: D = float32,
    layout: Layout = Strided,
    device: Device = CPU,
    requiresGrad: Boolean = false
): Tensor[D] = Tensor(
  torchNative.torch_eye(n, NativeConverters.tensorOptions(dtype, layout, device, requiresGrad))
)
// def empty(size: Long*): Tensor[D] = Tensor(torchNative.torch_empty(size*))

/** Returns a tensor filled with uninitialized data. */
def empty[D <: DType](
    size: Seq[Int],
    dtype: D = float32,
    layout: Layout = Strided,
    device: Device = CPU,
    requiresGrad: Boolean = false,
    pinMemory: Boolean = false,
    memoryFormat: MemoryFormat = Contiguous
): Tensor[D] =
  Tensor(
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
  Tensor(
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
    (key.toStringRef().getString(), Tensor[DType](value.toTensor().clone()))
  })

def pickleLoad(path: Path): Map[String, Tensor[DType]] =
  val data: Array[Byte] = Files.readAllBytes(path)
  pickleLoad(data)

def pickle_save(tensors: SeqMap[String, Tensor[DType]]) =
  tensors.map { (k, v) =>
    (IValue(k), IValue(v.native))
  }

/** Returns a tensor filled with random numbers from a uniform distribution on the interval `[0,1)`
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
  Tensor(
    torchNative.torch_rand(
      size.toArray.map(_.toLong),
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
    )
  )

/** Returns a tensor with the same size as `input` that is filled with random numbers from a uniform
  * distribution on the interval $[0, 1)$.
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

def randn[D <: FloatNN](
    size: Seq[Int],
    dtype: D = float32,
    layout: Layout = Strided,
    device: Device = CPU,
    requiresGrad: Boolean = false
): Tensor[D] =
  Tensor(
    torchNative.torch_rand(
      size.toArray.map(_.toLong),
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
    )
  )

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
  Tensor(
    torchNative.torch_randperm(
      n,
      NativeConverters.tensorOptions(dtype, layout, device, requiresGrad, pinMemory)
    )
  )

private def xLike[D <: DType, D2 <: DType | Derive](
    input: Tensor[D],
    dtype: D2,
    layout: Layout | Derive,
    device: Device | Derive,
    requiresGrad: Boolean,
    memoryFormat: MemoryFormat,
    nativeFn: (
        pytorch.Tensor,
        pytorch.TensorOptions,
        pytorch.MemoryFormatOptional
    ) => pytorch.Tensor
): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
  val derivedDType = dtype match
    case _: Derive => input.dtype
    case d: DType  => d
  val derivedLayout = layout match
    case _: Derive => input.layout
    case l: Layout => l
  val derivedDevice = device match
    case _: Derive => input.device
    case d: Device => d
  Tensor(
    nativeFn(
      input.native,
      NativeConverters.tensorOptions(derivedDType, derivedLayout, derivedDevice, requiresGrad),
      new MemoryFormatOptional(memoryFormat.toNative)
    )
  )

// End Creation Ops

// Indexing, Slicing, Joining, Mutating Ops

def cat[D <: DType](tensors: Seq[Tensor[D]], dim: Int = 0): Tensor[D] = Tensor(
  torchNative.cat(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)), dim.toLong)
)

// TODO dsplit
// TODO column_stack
// TODO dstack
// TODO gather
// TODO hsplit
// TODO hstack
// TODO index_add
// TODO index_copy
// TODO index_reduce
// TODO index_select
// TODO masked_select
// TODO movedim
// TODO moveaxis
// TODO narrow
// TODO narrow_copy
// TODO nonzero
// TODO permute
// TODO reshape
// TODO select
// TODO scatter
// TODO diagonal_scatter
// TODO select_scatter
// TODO slice_scatter
// TODO scatter_add
// TODO scatter_reduce
// TODO split
// TODO squeeze

/** Concatenates a sequence of tensors along a new dimension.
  *
  * All tensors need to be of the same size.
  */
def stack[D <: DType](tensors: Seq[Tensor[D]], dim: Int = 0): Tensor[D] = Tensor(
  torchNative.stack(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)), dim)
)

// TODO swapaxes
// TODO swapdims
// TODO t
// TODO take
// TODO take_along_dim
// TODO tensor_split
// TODO tile
// TODO transpose
// TODO unbind
// TODO unsqueeze
// TODO vsplit
// TODO vstack
// TODO where

// End Indexing, Slicing, Joining, Mutating Ops

// Math operations

// Pointwise Ops

/** Computes the absolute value of each element in `input`. */
def abs[D <: DType](input: Tensor[D]) = Tensor(torchNative.abs(input.native))

/** Computes the inverse cosine of each element in `input`. */
def acos[D <: DType](input: Tensor[D]) = Tensor(torchNative.acos(input.native))

/** Returns a new tensor with the inverse hyperbolic cosine of the elements of `input` . */
def acosh[D <: DType](input: Tensor[D]) = Tensor(torchNative.acosh(input.native))

/** Adds `other` to `input`. */
def add[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.add(input.native, other.native))

/** Adds `other` to `input`. */
def add[D <: DType, S <: ScalaType](
    input: Tensor[D],
    other: S
): Tensor[Promoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.add(input.native, toScalar(other)))

/** Performs the element-wise division of tensor1 by tensor2, multiplies the result by the scalar
  * value and adds it to input.
  */
def addcdiv[D <: DType, D2 <: DType, D3 <: DType](
    input: Tensor[D],
    tensor1: Tensor[D2],
    tensor2: Tensor[D3],
    value: ScalaType
): Tensor[Promoted[D, Promoted[D2, D3]]] =
  Tensor(torchNative.addcdiv(input.native, tensor1.native, tensor2.native, toScalar(value)))

/** Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by the
  * scalar value and adds it to input.
  */
def addcmul[D <: DType, D2 <: DType, D3 <: DType](
    input: Tensor[D],
    tensor1: Tensor[D2],
    tensor2: Tensor[D3],
    value: ScalaType
): Tensor[Promoted[D, Promoted[D2, D3]]] =
  Tensor(torchNative.addcmul(input.native, tensor1.native, tensor2.native, toScalar(value)))

/** Computes the element-wise angle (in radians) of the given `input` tensor. */
def angle[D <: DType](input: Tensor[D]): Tensor[ComplexToReal[D]] =
  Tensor(torchNative.angle(input.native))

/** Returns a new tensor with the arcsine of the elements of `input`. */
def asin[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.asin(input.native))

/** Returns a new tensor with the inverse hyperbolic sine of the elements of `input`. */
def asinh[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.asinh(input.native))

/** Returns a new tensor with the arctangent of the elements of `input`. */
def atan[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.atan(input.native))

/** Returns a new tensor with the inverse hyperbolic tangent of the elements of `input`. */
def atanh[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.atanh(input.native))

/** Element-wise arctangent of (input / other) with consideration of the quadrant. Returns a new
  * tensor with the signed angles in radians between vector (other, input) and vector (1, 0). (Note
  * that other, the second parameter, is the x-coordinate, while input, the first parameter, is the
  * y-coordinate.)
  */
def atan2[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.atan2(input.native, other.native))

/** Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or
  * Boolean types. For bool tensors, it computes the logical NOT.
  */
def bitwiseNot[D <: BitwiseNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_not(input.native))

/** Computes the bitwise AND of `input` and `other`. For bool tensors, it computes the logical AND.
  */
def bitwiseAnd[D <: BitwiseNN](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_and(input.native, other.native))

/** Computes the bitwise OR of `input` and `other`. For bool tensors, it computes the logical OR.
  */
def bitwiseOr[D <: BitwiseNN](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_or(input.native, other.native))

/** Computes the bitwise XOR of `input` and `other`. For bool tensors, it computes the logical XOR.
  */
def bitwiseXor[D <: BitwiseNN](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_xor(input.native, other.native))

/** Computes the left arithmetic shift of `input` by `other` bits. */
def bitwiseLeftShift[D <: IntNN](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_left_shift(input.native, other.native))

/** Computes the right arithmetic s\hift of `input` by `other` bits. */
def bitwiseRightShift[D <: IntNN](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.bitwise_right_shift(input.native, other.native))

/** Returns a new tensor with the ceil of the elements of `input`, the smallest integer greater than
  * or equal to each element.
  */
def ceil[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.ceil(input.native))

/** Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min
  * and max, respectively, this returns: `min(max(input, min_value), max_value)` If min is None,
  * there is no lower bound. Or, if max is None there is no upper bound.
  */
def clamp[D <: NumericNN](
    input: Tensor[D],
    min: Option[Tensor[D]],
    max: Option[Tensor[D]]
): Tensor[D] =
  Tensor(torchNative.clamp(input.native, toOptional(min), toOptional(max)))

/** Computes the element-wise conjugate of the given input tensor. If input has a non-complex dtype,
  * this function just returns input.
  */
def conjPhysical[D <: NumericNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.conj_physical(input.native))

/** Create a new floating-point tensor with the magnitude of input and the sign of other,
  * elementwise.
  */
// TODO
// def copysign[D <: DType](input: Tensor[D]): Tensor[D] =
//   Tensor(torchNative.copysign(input.native))

/** Returns a new tensor with the cosine of the elements of `input`. */
def cos[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.cos(input.native))

/** Returns a new tensor with the hyperbolic cosine of the elements of `input`. */
def cosh[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.cosh(input.native))

/** Returns a new tensor with each of the elements of `input` converted from angles in degrees to
  * radians.
  */
def deg2rad[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.deg2rad(input.native))

/** Divides each element of the input `input` by the corresponding element of `other`. */
def div[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[D] =
  Tensor(torchNative.div(input.native, other.native))

export torch.special.digamma
export torch.special.erf
export torch.special.erfc
export torch.special.erfinv

/** Returns a new tensor with the exponential of the elements of the input tensor `input`. */
def exp[D <: RealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.exp(input.native))

export torch.special.exp2
export torch.special.expm1

/** Returns a new tensor with the data in `input` fake quantized per channel using `scale`,
  * `zero_point`, `quant_min` and `quant_max`, across the channel specified by `axis`.
  */
// TODO Fix pytorch docs to add `axis` input
def fakeQuantizePerChannelAffine(
    input: Tensor[Float32],
    scale: Tensor[Float32],
    zeroPoint: Tensor[Int32 | Float16 | Float32],
    axis: Long,
    quantMin: Long,
    quantMax: Long
): Tensor[Float32] =
  Tensor(
    torchNative.fake_quantize_per_channel_affine(
      input.native,
      scale.native,
      zeroPoint.native,
      axis,
      quantMin,
      quantMax
    )
  )

/** Returns a new tensor with the data in `input` fake quantized using `scale`, `zero_point`,
  * `quant_min` and `quant_max`.
  */
def fakeQuantizePerTensorAffine(
    input: Tensor[Float32],
    scale: Tensor[Float32],
    zeroPoint: Tensor[Int32],
    quantMin: Long,
    quantMax: Long
): Tensor[Float32] =
  Tensor(
    torchNative.fake_quantize_per_tensor_affine(
      input.native,
      scale.native,
      zeroPoint.native,
      quantMin,
      quantMax
    )
  )

def fakeQuantizePerTensorAffine(
    input: Tensor[Float32],
    scale: Double,
    zeroPoint: Long,
    quantMin: Long,
    quantMax: Long
): Tensor[Float32] =
  Tensor(
    torchNative.fake_quantize_per_tensor_affine(input.native, scale, zeroPoint, quantMin, quantMax)
  )

// TODO torch.fix // Alias for torch.trunc

/** Raises `input` to the power of `exponent`, elementwise, in double precision. If neither input is
  * complex returns a `torch.float64` tensor, and if one or more inputs is complex returns a
  * `torch.complex128` tensor.
  */
def floatPower[D <: DType, D2 <: DType](
    input: Tensor[D],
    exponent: Tensor[D2]
): Tensor[ComplexPromoted[D, D2]] =
  Tensor(torchNative.float_power(input.native, exponent.native))

def floatPower[D <: DType, S <: ScalaType](
    input: S,
    exponent: Tensor[D]
): Tensor[ComplexPromoted[ScalaToDType[S], D]] =
  Tensor(torchNative.float_power(toScalar(input), exponent.native))

def floatPower[D <: DType, S <: ScalaType](
    input: Tensor[D],
    exponent: ScalaType
): Tensor[ComplexPromoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.float_power(input.native, toScalar(exponent)))

/** Returns a new tensor with the floor of the elements of `input`, the largest integer less than or
  * equal to each element.
  */
def floor[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.floor(input.native))

/** Computes `input` divided by `other`, elementwise, and floors the result. */
def floorDivide[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: TensorOrReal[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(
    (input, other) match
      case (input: Tensor[D], other: Tensor[D2]) =>
        torchNative.floor_divide(input.native, other.native)
      case (input: Tensor[D], other: Real) =>
        torchNative.floor_divide(input.native, toScalar(other))
  )

/** Applies C++’s `std::fmod` entrywise. The result has the same sign as the dividend `input` and
  * its absolute value is less than that of `other`.
  */
// NOTE: When the divisor is zero, returns NaN for floating point dtypes on both CPU and GPU; raises RuntimeError for integer division by zero on CPU; Integer division by zero on GPU may return any value.
def fmod[D <: RealNN](input: Tensor[D], other: TensorOrReal[D]): Tensor[D] =
  Tensor(
    other match
      case (other: Tensor[D]) =>
        torchNative.fmod(input.native, other.native)
      case (other: Real) =>
        torchNative.fmod(input.native, toScalar(other))
  )

/** Computes the fractional portion of each element in `input`. */
def frac[D <: FloatNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.frac(input.native))

/** Decomposes `input` into `mantissa` and `exponent` tensors such that `input = mantissa * (2 **
  * exponent)` The range of mantissa is the open interval (-1, 1).
  */
def frexp[D <: FloatNN](input: Tensor[D]): (Tensor[FloatPromoted[D]], Tensor[Int32]) =
  val nativeTuple = torchNative.frexp(input.native)
  (Tensor(nativeTuple.get0), new Int32Tensor(nativeTuple.get1))

// TODO implement
/** */
// def gradient[D <: DType](input: Tensor[D]): Tensor[D] =
//   Tensor(torchNative.???)

/** Returns a new tensor containing imaginary values of the `input` tensor. The returned tensor and
  * `input` share the same underlying storage.
  */
def imag[D <: ComplexNN](input: Tensor[D]): Tensor[ComplexToReal[D]] =
  Tensor(torchNative.imag(input.native))

/** Multiplies `input` by 2 ** `other`. */
def ldexp[D <: DType](input: Tensor[D], other: Tensor[D]): Tensor[D] =
  Tensor(torchNative.ldexp(input.native, other.native))

/** Does a linear interpolation of two tensors `start` (given by `input`) and `end` (given by
  * `other`) based on a scalar or tensor weight and returns the resulting out tensor. out = start +
  * weight × (end − start)
  */
def lerp[D <: DType](
    input: Tensor[D],
    other: Tensor[D],
    weight: Tensor[D] | Float | Double
): Tensor[D] =
  Tensor(
    weight match
      case weight: Tensor[D] => torchNative.lerp(input.native, other.native, weight.native)
      case weight: Float     => torchNative.lerp(input.native, other.native, toScalar(weight))
      case weight: Double    => torchNative.lerp(input.native, other.native, toScalar(weight))
  )

/** Computes the natural logarithm of the absolute value of the gamma function on `input`. */
def lgamma[D <: RealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.lgamma(input.native))

/** Returns a new tensor with the natural logarithm of the elements of `input`. */
def log[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log(input.native))

/** Returns a new tensor with the logarithm to the base 10 of the elements of `input`. */
def log10[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log10(input.native))

/** Returns a new tensor with the natural logarithm of (1 + input). */
def log1p[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log1p(input.native))

/** Returns a new tensor with the logarithm to the base 2 of the elements of `input`. */
def log2[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.log2(input.native))

/** Logarithm of the sum of exponentiations of the inputs. Calculates pointwise log `log(e**x +
  * e**y)`. This function is useful in statistics where the calculated probabilities of events may
  * be so small as to exceed the range of normal floating point numbers. In such cases the logarithm
  * of the calculated probability is stored. This function allows adding probabilities stored in
  * such a fashion. This op should be disambiguated with `torch.logsumexp()` which performs a
  * reduction on a single tensor.
  */
def logaddexp[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.logaddexp(input.native, other.native))

/** Logarithm of the sum of exponentiations of the inputs in base-2. Calculates pointwise `log2(2**x
  * + 2**y)`. See torch.logaddexp() for more details.
  */
def logaddexp2[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.logaddexp2(input.native, other.native))

/** Computes the element-wise logical AND of the given input tensors. Zeros are treated as False and
  * nonzeros are treated as True.
  */
def logicalAnd[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
  Tensor(torchNative.logical_and(input.native, other.native))

/** Computes the element-wise logical NOT of the given input tensor. TODO If not specified, the
  * output tensor will have the bool dtype. If the input tensor is not a bool tensor, zeros are
  * treated as False and non-zeros are treated as True.
  */
def logicalNot[D <: RealNN](input: Tensor[D]): Tensor[Bool] =
  Tensor(torchNative.logical_not(input.native))

/** Computes the element-wise logical OR of the given input tensors. Zeros are treated as False and
  * nonzeros are treated as True.
  */
def logicalOr[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
  Tensor(torchNative.logical_or(input.native, other.native))

/** Computes the element-wise logical XOR of the given input tensors. Zeros are treated as False and
  * nonzeros are treated as True.
  */
def logicalXor[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Bool] =
  Tensor(torchNative.logical_or(input.native, other.native))

export torch.special.logit

/** Given the legs of a right triangle, return its hypotenuse. */
def hypot[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.hypot(input.native, other.native))

export torch.special.i0
export torch.special.igamma
export torch.special.igammac

/** Multiplies input by other. */
def mul[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.mul(input.native, other.native))

export torch.special.mvlgamma

/** Replaces NaN, positive infinity, and negative infinity values in `input` with the values
  * specified by nan, posinf, and neginf, respectively. By default, NaNs are replaced with zero,
  * positive infinity is replaced with the greatest finite value representable by input’s dtype, and
  * negative infinity is replaced with the least finite value representable by input’s dtype.
  */
def nanToNum[D <: FloatNN](
    input: Tensor[D],
    nan: Option[Double] = None,
    posinf: Option[Double],
    neginf: Option[Double]
): Tensor[D] =
  Tensor(
    torchNative.nan_to_num(input.native, toOptional(nan), toOptional(posinf), toOptional(neginf))
  )

/** Returns a new tensor with the negative of the elements of `input`. */
def neg[D <: NumericNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.neg(input.native))

/** Return the next floating-point value after `input` towards `other`, elementwise. */
def nextafter[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2]
): Tensor[FloatPromoted[Promoted[D, D2]]] =
  Tensor(torchNative.nextafter(input.native, other.native))

export torch.special.polygamma

/** Returns input. Normally throws a runtime error if input is a bool tensor in pytorch. */
def positive[D <: NumericNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.positive(input.native))

/** Takes the power of each element in `input` with exponent and returns a tensor with the result.
  * `exponent` can be either a single float number or a Tensor with the same number of elements as
  * input.
  */
// TODO handle Scalar `input`
def pow[D <: DType, D2 <: DType](
    input: Tensor[D],
    exponent: TensorOrReal[D2]
): Tensor[FloatPromoted[D]] =
  Tensor(
    (input, exponent) match
      case (input: Tensor[D], exponent: Tensor[D2]) =>
        torchNative.pow(input.native, exponent.native)
      case (input: Tensor[D], exponent: Real) =>
        torchNative.pow(input.native, toScalar(exponent))
  )

// TODO quantized_batch_norm
// TODO quantized_max_pool1d
// TODO quantized_max_pool2d

/** Returns a new tensor with each of the elements of `input` converted from angles in radians to
  * degrees.
  */
def rad2Deg[D <: RealNN | Bool](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.rad2deg(input.native))

/** Returns a new tensor containing real values of the self tensor. The returned tensor and self
  * share the same underlying storage.
  */
def real[D <: DType](input: Tensor[D]): Tensor[ComplexToReal[D]] =
  Tensor(torchNative.real(input.native))

/** Returns a new tensor with the reciprocal of the elements of `input` */
def reciprocal[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.reciprocal(input.native))

/** Computes Python’s modulus operation entrywise. The result has the same sign as the divisor
  * `other` and its absolute value is less than that of `other`.
  */
// TODO handle Scalar `input`
def remainder[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: TensorOrReal[D2]
): Tensor[FloatPromoted[D]] =
  Tensor(
    (input, other) match
      case (input: Tensor[D], other: Tensor[D2]) =>
        torchNative.remainder(input.native, other.native)
      case (input: Tensor[D], other: Real) =>
        torchNative.remainder(input.native, toScalar(other))
  )

/** Rounds elements of `input` to the nearest integer. If decimals is negative, it specifies the
  * number of positions to the left of the decimal point.
  */
def round[D <: NumericNN](input: Tensor[D], decimals: Long = 0): Tensor[D] =
  Tensor(torchNative.round(input.native, decimals))

/** Returns a new tensor with the reciprocal of the square-root of each of the elements of `input`.
  */
def rsqrt[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.rsqrt(input.native))

export torch.special.sigmoid

/** Returns a new tensor with the signs of the elements of `input`. */
def sign[D <: RealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.sign(input.native))

/** This function is an extension of `torch.sign()` to complex tensors. It computes a new tensor
  * whose elements have the same angles as the corresponding elements of `input` and absolute values
  * (i.e. magnitudes) of one for complex tensors and is equivalent to torch.sign() for non-complex
  * tensors.
  */
def sgn[D <: DType](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.sgn(input.native))

/** Tests if each element of `input`` has its sign bit set or not. */
def signbit[D <: RealNN](input: Tensor[D]): Tensor[Bool] =
  Tensor(torchNative.signbit(input.native))

/** Returns a new tensor with the sine of the elements of `input`. */
def sin[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.sin(input.native))

export torch.special.sinc

/** Returns a new tensor with the hyperbolic sine of the elements of `input`. */
def sinh[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.sinh(input.native))

// TODO softmax

export torch.nn.functional.softmax

/** Returns a new tensor with the square-root of the elements of `input`. */
def sqrt[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.sqrt(input.native))

/** Returns a new tensor with the square of the elements of `input`. */
def square[D <: RealNN](input: Tensor[D]): Tensor[NumericPromoted[D]] =
  Tensor(torchNative.square(input.native))

/** Subtracts `other`, scaled by `alpha`, from `input`. */
def sub[D <: DType, D2 <: DType](input: Tensor[D], other: Tensor[D2]): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.sub(input.native, other.native))

def sub[D <: DType, D2 <: DType](
    input: Tensor[D],
    other: Tensor[D2],
    alpha: ScalaType
): Tensor[Promoted[D, D2]] =
  Tensor(torchNative.sub(input.native, other.native, toScalar(alpha)))

def sub[D <: DType, S <: ScalaType](
    input: Tensor[D],
    other: S,
    alpha: ScalaType
): Tensor[Promoted[D, ScalaToDType[S]]] =
  Tensor(torchNative.sub(input.native, toScalar(other), toScalar(alpha)))

/** Returns a new tensor with the tangent of the elements of `input`. */
def tan[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.tan(input.native))

/** Returns a new tensor with the hyperbolic tangent of the elements of `input`. */
def tanh[D <: RealNN](input: Tensor[D]): Tensor[FloatPromoted[D]] =
  Tensor(torchNative.tanh(input.native))

// TODO true_divide

/** Returns a new tensor with the truncated integer values of the elements of `input`. */
def trunc[D <: NumericRealNN](input: Tensor[D]): Tensor[D] =
  Tensor(torchNative.trunc(input.native))

// TODO xlogy

// End Pointwise Ops

// Comparison Ops

def allclose(
    input: Tensor[?],
    other: Tensor[?],
    rtol: Double = 1e-05,
    atol: Double = 1e-08,
    equalNan: Boolean = false
) =
  torchNative.allclose(input.native, other.native, rtol, atol, equalNan)

// End Comparison Ops

// End Math operations

def matmul[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
  t1.matmul(t2)

def manualSeed(seed: Long) = torchNative.manual_seed(seed)

/** Disable gradient calculation for [[op]].
  *
  * Disabling gradient calculation is useful for inference, when you are sure that you will not call
  * `Tensor.backward()`. It will reduce memory consumption for computations that would otherwise
  * have `requiresGrad=true`.
  *
  * In this mode, the result of every computation will have `requiresGrad=false`, even when the
  * inputs have `requiresGrad=true`.
  *
  * This context manager is thread local; it will not affect computation in other threads.
  *
  * @param op
  */
def noGrad[A](op: => A): A = {
  import org.bytedeco.pytorch.NoGradGuard
  Using.resource(NoGradGuard()) { _ =>
    op
  }
}

def setNumThreads(threads: Int): Unit = torchNative.set_num_threads(threads)
