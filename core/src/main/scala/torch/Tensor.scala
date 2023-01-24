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

import org.bytedeco.javacpp.{
  BoolPointer,
  BytePointer,
  DoublePointer,
  FloatPointer,
  IntPointer,
  LongPointer,
  ShortPointer
}
import org.bytedeco.javacpp.indexer.{Indexer, IntIndexer, LongIndexer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LongOptional, Scalar, TensorIndexArrayRef}
import org.bytedeco.pytorch.global.torch as torchNative
import Tensor.*
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.NoGradGuard
import ScalarUtils.toScalar

import java.nio.{
  Buffer,
  ByteBuffer,
  CharBuffer,
  DoubleBuffer,
  FloatBuffer,
  IntBuffer,
  LongBuffer,
  ShortBuffer
}
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import scala.annotation.{targetName, unused}
import org.bytedeco.pytorch.global.torch.DeviceType
import internal.NativeConverters.toOptional
import spire.math.{Complex, UByte}

import scala.reflect.Typeable
import internal.NativeConverters
import torch.Device.CPU
import torch.Layout.Strided
import org.bytedeco.pytorch.ByteArrayRef
import org.bytedeco.pytorch.ShortArrayRef
import org.bytedeco.pytorch.BoolArrayRef
import org.bytedeco.pytorch.IntArrayRef
import org.bytedeco.pytorch.LongArrayRef
import org.bytedeco.pytorch.FloatArrayRef
import org.bytedeco.pytorch.DoubleArrayRef

case class TensorTuple[D <: DType](
    values: Tensor[D],
    indices: Tensor[Int64]
)

/** A [[torch.Tensor]] is a multi-dimensional matrix containing elements of a single data type. */
sealed abstract class Tensor[D <: DType]( /* private[torch]  */ val native: pytorch.Tensor):

  def ==(other: ScalaType): Tensor[Bool] = eq(other)

  @targetName("add")
  def +[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = Tensor(
    native.add(toScalar(s))
  )

  @targetName("add")
  def +[D2 <: DType](t: Tensor[D2]): Tensor[Promoted[D, D2]] = Tensor(native.add(t.native))

  // TODO add typelevel casting rules. I.e. An integral output tensor cannot accept a floating point tensor.
  // https://github.com/pytorch/pytorch/blob/041edeeecb75f3c110605d7311fa46abe1c62ea9/c10/core/ScalarType.h
  // https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
  def +=[D2 <: DType](t: Tensor[D2]): this.type =
    native.add_(t.native)
    this

  def -[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = Tensor(
    native.sub(toScalar(s))
  )

  def -[D2 <: DType](t: Tensor[D2]): Tensor[Promoted[D, D2]] = Tensor(native.sub(t.native))

  def -=[D2 <: DType](t: Tensor[D2]): this.type =
    native.sub_(t.native)
    this

  @targetName("mul")
  def *[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = Tensor(
    native.mul(toScalar(s))
  )

  @targetName("mul")
  def *[D2 <: DType](t: Tensor[D2]): Tensor[Promoted[D, D2]] = Tensor(native.mul(t.native))

  def *=[D2 <: DType](t: Tensor[D2]): this.type =
    native.mul_(t.native)
    this

  def /[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = Tensor(
    native.div(toScalar(s))
  )

  def /[D2 <: DType](t: Tensor[D2]): Tensor[Promoted[D, D2]] = Tensor(native.div(t.native))

  def /=[D2 <: DType](t: Tensor[D2]): this.type =
    native.div_(t.native)
    this

  def apply[T <: Boolean | Long: ClassTag](
      indices: (Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int64] | Seq[T] |
        None.type)*
  ): Tensor[D] = index(indices*)

  /** Computes the absolute value of each element. */
  def abs: Tensor[D] = Tensor(native.abs())

  def acos: Tensor[D] = Tensor(native.acos())

  /** Tests if all elements of this tensor evaluate to `true`. */
  def all: Tensor[Bool] = Tensor(native.all())

  def any: Tensor[Bool] = Tensor(native.any())

  /** Tests if any element of this tensor evaluates to `true`. */
  def any(dim: Int, keepdim: Boolean = true): Tensor[Bool] = Tensor(native.any(dim, keepdim))

  /** Returns the indices of the maximum value of all elements in the tensor.
    *
    * This is the second value returned by torch.max(). See its documentation for the exact
    * semantics of this method.
    *
    * Example:
    * ```scala sc:nocompile
    * val a = torch.rand(Seq(1, 3))
    * println(a.argmax())
    * // tensor dtype=float32, shape=[1] 2
    * ```
    *
    * @param dim
    * @param keepdim
    * @return
    */
  def argmax(dim: Long | Option[Long] = None, keepdim: Boolean = false): Tensor[Int64] = Tensor(
    native.argmax(NativeConverters.toOptional(dim), keepdim)
  )

  def backward(): Unit = native.backward()

  /** Returns a new Tensor, detached from the current graph.
    *
    * The result will never require gradient.
    *
    * This method also affects forward mode AD gradients and the result will never have forward mode
    * AD gradients.
    */
  def detach(): Tensor[D] = Tensor(native.detach())

  /** Returns a copy of `input`.
    *
    * @note
    *   This function is differentiable, so gradients will flow back from the result of this
    *   operation to `input`. To create a tensor without an autograd relationship to `input` see
    *   `Tensor.detach`.
    */
  def clone(memoryFormat: MemoryFormat = MemoryFormat.Preserve): Tensor[D] =
    Tensor(native.clone(memoryFormat.toNativeOptional))

  def contiguous: Tensor[D] = Tensor(native.contiguous())

  /** Copies the elements from `src` into this tensor and returns this.
    *
    * The `src` tensor must be broadcastable with the self tensor. It may be of a different data
    * type or reside on a different device.
    *
    * @param src
    *   the source tensor to copy from
    * @param nonBlocking
    *   if `true` and this copy is between CPU and GPU, the copy may occur asynchronously with
    *   respect to the host. For other cases, this argument has no effect.
    */
  def copy_(src: Tensor[?], nonBlocking: Boolean = false): this.type =
    native.copy_(src.native, nonBlocking)
    this

  def device: Device = Device(native.device())

  def dim: Int = native.dim().toInt

  def dtype: D // = DType.fromScalarType(native.dtype().toScalarType)
  // def dtype: T = deriveDType[T]

  /** Computes element-wise equality */
  def eq(other: ScalaType): Tensor[Bool] = Tensor(native.eq(toScalar(other)))

  /** Computes element-wise equality
    *
    * The argument can be a tensor whose shape is broadcastable with this tensor.
    */
  def eq(other: Tensor[?]): Tensor[Bool] = Tensor(native.eq(other.native))

  override def equals(that: Any): Boolean =
    that match
      case t: Tensor[?] if dtype == t.dtype => native.equal(t.native)
      case _                                => false

  /** True if `other` has the same size and elements as this tensor, false otherwise. */
  def equal(other: Tensor[D]): Boolean = native.equal(other.native)

  def flatten: Tensor[D] = Tensor(native.flatten())

  def flatten(startDim: Int = 0, endDim: Int = -1): Tensor[D] = Tensor(
    native.flatten(startDim, endDim)
  )

  /** This function returns an undefined tensor by default and returns a defined tensor the first
    * time a call to backward() computes gradients for this Tensor. The attribute will then contain
    * the gradients computed and future calls to backward() will accumulate (add) gradients into it.
    */
  def grad: Tensor[D | Undefined] = Tensor(native.grad())

  def isContiguous: Boolean = native.is_contiguous()

  def isCuda: Boolean = native.is_cuda()

  def isQuantized: Boolean = native.is_quantized()

  def isnan: Tensor[Bool] = Tensor(native.isnan())

  def isNonzero: Boolean = native.is_nonzero()

  // TODO override in subclasses instead?
  def item: DTypeToScala[D] =
    import ScalarType.*
    val out = native.dtype().toScalarType().intern() match
      case Byte                   => native.item_byte()
      case Char                   => UByte(native.item_byte())
      case Short                  => native.item_short()
      case Int                    => native.item_int()
      case Long                   => native.item_long()
      case Half                   => native.item().toHalf.asFloat()
      case Float                  => native.item_float()
      case Double                 => native.item_double()
      case ComplexHalf            => ??? // TODO how to access complex scalar values?
      case ComplexFloat           => ???
      case ComplexDouble          => ???
      case Bool                   => native.item().toBool
      case QInt8                  => native.item_byte()
      case QUInt8                 => native.item_short()
      case QInt32                 => native.item_int()
      case BFloat16               => native.item().toBFloat16.asFloat()
      case QUInt4x2               => ???
      case QUInt2x4               => ???
      case Undefined | NumOptions => ???
    out.asInstanceOf[DTypeToScala[D]]

  def layout: Layout = Layout.fromNative(native.layout())

  def matmul[D2 <: DType](u: Tensor[D2]): Tensor[Promoted[D, D2]] =
    Tensor[Promoted[D, D2]](native.matmul(u.native))

  /** Returns the maximum value of all elements in the ``input`` tensor. */
  def max(): Tensor[Int64] = Tensor(native.max())

  /** Returns a tuple ``(values, indices)`` where ``values`` is the maximum value of each row of the
    * `input` tensor in the given dimension `dim`. And ``indices`` is the index location of each
    * maximum value found (argmax).
    *
    * If ``keepdim`` is ``true``, the output tensors are of the same size as ``input`` except in the
    * dimension ``dim`` where they are of size 1. Otherwise, ``dim`` is squeezed (see
    * :func:`torch.squeeze`), resulting in the output tensors having 1 fewer dimension than
    * ``input``.
    *
    * @note
    *   If there are multiple maximal values in a reduced row then the indices of the first maximal
    *   value are returned.
    */
  def max(dim: Long, keepdim: Boolean = false): TensorTuple[D] =
    val nativeTuple = native.max(dim, keepdim)
    TensorTuple(values = Tensor[D](nativeTuple.get0), indices = new Int64Tensor(nativeTuple.get1))

  def maximum[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] =
    Tensor[Promoted[D, D2]](native.maximum(other.native))

  def mean: Tensor[D] = Tensor(native.mean())

  def min(): Tensor[Int64] = Tensor[Int64](native.min())

  def minimum[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] =
    Tensor[Promoted[D, D2]](native.minimum(other.native))

  /** Returns the total number of elements in the input tensor. */
  def numel: Long = native.numel()

  def permute(dims: Long*): Tensor[D] = Tensor(native.permute(dims*))

  def pow(exponent: Double): Tensor[D] = Tensor(native.pow(Scalar.apply(exponent)))

  def prod[D <: DType](dtype: D = this.dtype) = Tensor(native.prod())

  def reshape(shape: Long*): Tensor[D] = Tensor(native.reshape(shape*))

  def shape: Seq[Long] = size

  def square = Tensor(native.square())

  def squeeze: Tensor[D] = Tensor(native.squeeze())

  def size: Seq[Long] = ArraySeq.unsafeWrapArray(native.sizes.vec.get)

  def std: Tensor[D] = Tensor[D](native.std())

  /** Returns the sum of all elements of this tensor. */
  def sum: Tensor[Sum[D]] = Tensor(native.sum())

  def zero(): Unit = native.zero_()

  def index[T <: Boolean | Long: ClassTag](
      indices: (Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int64] | Seq[T] |
        None.type)*
  ): Tensor[D] =
    val nativeIndices: Seq[pytorch.TensorIndex] =
      for (i <- indices) yield i match
        case None =>
          new pytorch.TensorIndex()
        case t: Tensor[?] =>
          new pytorch.TensorIndex(t.native)
        case singleton: Int =>
          new pytorch.TensorIndex(singleton)
        case singleton: Long =>
          new pytorch.TensorIndex(singleton)
        case Slice(start, end, step) =>
          new pytorch.TensorIndex(
            new pytorch.Slice(toOptional(start), toOptional(end), toOptional(step))
          )
        case s: Seq[T] @unchecked => new pytorch.TensorIndex(Tensor[T](s).native)
        // TODO remaining index types, see https://pytorch.org/cppdocs/notes/tensor_indexing.html
    val ref = new pytorch.TensorIndexArrayRef(new pytorch.TensorIndexVector(nativeIndices.toArray*))
    Tensor(native.index(ref))

  def requiresGrad: Boolean = native.requires_grad()

  def requiresGrad_=(requiresGrad: Boolean): Unit = native.requires_grad_(requiresGrad)

  def take(indices: Tensor[Int64]): Tensor[D] = Tensor(native.take(indices.native))

  def takeAlongDim(indices: Tensor[Int64], dim: Int) =
    native.take_along_dim(indices.native, toOptional(dim))

  // TODO support memory_format
  /** Performs Tensor dtype and/or device conversion. */
  def to[U <: DType](
      device: Device = this.device,
      dtype: U = this.dtype,
      nonBlocking: Boolean = false,
      copy: Boolean = false
  ): Tensor[U] =
    val targetDType = dtype.toScalarType
    if dtype == this.dtype && device == this.device && !copy then this.asInstanceOf[Tensor[U]]
    else if device == this.device then
      Tensor(
        native.to(
          targetDType,
          nonBlocking,
          copy,
          pytorch.MemoryFormatOptional(torchNative.MemoryFormat.Preserve)
        )
      )
    else
      Tensor(
        native.to(
          device.toNative,
          targetDType,
          nonBlocking,
          copy,
          pytorch.MemoryFormatOptional(torchNative.MemoryFormat.Preserve)
        )
      )

  def toBuffer: TypedBuffer[DTypeToScala[D]] =
    to(device = CPU).native.createBuffer[TypedBuffer[DTypeToScala[D]]]()

  def toArray: Array[DTypeToScala[D]] =

    val tensor = to(device = CPU)
    def writeArray[A: ClassTag, B <: Buffer](getElem: B => A): Array[A] =
      val a = new Array[A](numel.toInt)
      if numel > 0 then
        val buf = tensor.native.contiguous.createBuffer[B]
        var i = 0
        while i < a.length do
          a(i) = getElem(buf)
          i += 1
      a

    def writeRawArray[A <: ScalaType: ClassTag](
        get: (Array[A], TypedBuffer[A]) => TypedBuffer[A]
    ): Array[A] =
      val a = new Array[A](numel.toInt)
      if numel > 0 then get(a, tensor.native.contiguous.createBuffer[TypedBuffer[A]])
      a

    import ScalarType.*
    val out = tensor.native.dtype().toScalarType.intern() match
      case Byte         => writeRawArray[Byte]((a, b) => b.get(a))
      case Short        => writeRawArray[Short]((a, b) => b.get(a))
      case Int          => writeRawArray[Int]((a, b) => b.get(a))
      case Long         => writeRawArray[Long]((a, b) => b.get(a))
      case Float        => writeRawArray[Float]((a, b) => b.get(a))
      case Double       => writeRawArray[Double]((a, b) => b.get(a))
      case Char         => to(dtype = int16).toArray
      case Half         => to(dtype = float32).toArray
      case ComplexHalf  => to(dtype = complex64).toArray
      case ComplexFloat => writeArray[Complex[Float], FloatBuffer](b => Complex(b.get(), b.get()))
      case ComplexDouble =>
        writeArray[Complex[Double], DoubleBuffer](b => Complex(b.get(), b.get()))
      case Bool       => writeArray[Boolean, ByteBuffer](b => b.get > 0)
      case QInt8      => ???
      case QUInt8     => ???
      case QInt32     => ???
      case BFloat16   => to(dtype = float32).toArray
      case QUInt4x2   => ???
      case QUInt2x4   => ???
      case Undefined  => ???
      case NumOptions => ???
    out.asInstanceOf[Array[DTypeToScala[D]]]

  def toSeq: Seq[DTypeToScala[D]] = ArraySeq.unsafeWrapArray(toArray)

  /** Returns a summary of the contents of this tensor.
    *
    * @param maxEntries
    *   Maximum number of entries to show for each axis/dimension. If the size of an axis exceeds
    *   `maxEntries`, the output of that axis will be shortened to the first and last three
    *   elements. Defaults to `6`. Values below `6` are ignored.
    * @param flattened
    *   If `true`, the summary is flattened to one line. Otherwise, the summary may span multiple
    *   lines.
    * @param includeInfo
    *   If `true`, the data type and the shape of the tensor are explicitly included in the summary.
    *   Otherwise, they are not.
    * @return
    *   Tensor summary.
    */
  def summarize(
      maxEntries: Int = 6,
      flattened: Boolean = false,
      includeInfo: Boolean = true
  ): String =
    def format(x: Any): String =
      x match
        case x: Float  => "%1.4f".format(x)
        case x: Double => "%1.4f".format(x)
        case x         => x.toString

    def summarize(tensor: Tensor[D], maxEntries: Int): String =
      tensor.dim match
        case 0 => format(tensor.toSeq.head) // ScalarUtils.scalarToString(tensor.native.item)
        case 1 =>
          val slice =
            if tensor.numel <= math.max(maxEntries, 6) then tensor.toSeq.map(format)
            else
              val left = tensor(Slice(0, maxEntries / 2)).toSeq.map(format)
              val right = tensor(Slice(-maxEntries / 2)).toSeq.map(format)
              left ++ Seq("...") ++ right
          slice.mkString("[", ", ", "]")
        case _ =>
          val innerSummary = {
            def summarizeSlice(index: Int) = summarize(tensor(index), maxEntries)
            val sliceLen = tensor.size(0).toInt
            if sliceLen <= math.max(maxEntries, 6) then
              for (i <- 0 until sliceLen.toInt) yield summarizeSlice(i)
            else
              val start = for (i <- 0 until maxEntries / 2) yield summarizeSlice(i)
              val end = for (i <- sliceLen - maxEntries / 2 until sliceLen) yield summarizeSlice(i)
              (start :+ "...") ++ end
          }
          val padding = " " * (this.dim - tensor.dim + 1)
          val extraLine = if (!flattened && tensor.dim >= 3) "\n" else ""
          innerSummary.mkString("[", (if (!flattened) ",\n" else ", ") + extraLine + padding, "]")

    if dtype == undefined then "undefined tensor"
    else if includeInfo then
      s"dtype=${dtype.toString}, shape=${size.mkString("[", ", ", "]")}, device=${device.device} " +
        (if (!flattened) "\n" else ": ") + summarize(this, maxEntries)
    else summarize(this, maxEntries)

  def view(shape: Int*): Tensor[D] = Tensor(native.view(shape.map(_.toLong)*))

  override def toString: String =
    summarize() // s"dtype=${dtype.toString}, shape=${size.mkString("[", ", ", "]")}, device=${device.device} " //summarize()

  def requireNativeType(expected: ScalarType) = require(
    native.scalar_type().intern() == expected,
    s"Expected native tensor type $expected, got ${native.scalar_type().intern()}"
  )

sealed class Int8Tensor(native: pytorch.Tensor) extends Tensor[Int8](native) { /* 0, Byte */
  require(native.scalar_type().intern() == ScalarType.Byte)
  override def dtype: Int8 = int8
}

sealed class UInt8Tensor(native: pytorch.Tensor) extends Tensor[UInt8](native) { /* 1, Char */
  requireNativeType(ScalarType.Char)
  override def dtype: UInt8 = uint8
}
sealed class Int16Tensor(native: pytorch.Tensor) extends Tensor[Int16](native) { /* 2, Short */
  requireNativeType(ScalarType.Short)
  override def dtype: Int16 = int16
}
sealed class Int32Tensor(native: pytorch.Tensor) extends Tensor[Int32](native) { /* 3, Int */
  requireNativeType(ScalarType.Int)
  override def dtype: Int32 = int32
}
sealed class Int64Tensor(native: pytorch.Tensor) extends Tensor[Int64](native) { /* 4, Long */
  requireNativeType(ScalarType.Long)
  override def dtype: Int64 = int64
}
sealed class Float32Tensor(native: pytorch.Tensor) extends Tensor[Float32](native) { /* 5, Float */
  requireNativeType(ScalarType.Float)
  override def dtype: Float32 = float32
}
sealed class Float64Tensor(native: pytorch.Tensor) extends Tensor[Float64](native) { /* 6, Double */
  requireNativeType(ScalarType.Double)
  override def dtype: Float64 = float64
}
sealed class Complex32Tensor(native: pytorch.Tensor) extends Tensor[Complex32](native) { /* 7, ComplexHalf */
  requireNativeType(ScalarType.ComplexHalf)
  override def dtype: Complex32 = complex32
}
sealed class Complex64Tensor(native: pytorch.Tensor) extends Tensor[Complex64](native) { /* 8, ComplexFloat */
  requireNativeType(ScalarType.ComplexFloat)
  override def dtype: Complex64 = complex64
}
sealed class Complex128Tensor(native: pytorch.Tensor) extends Tensor[Complex128](native) { /* 9, ComplexDouble */
  requireNativeType(ScalarType.ComplexDouble)
  override def dtype: Complex128 = complex128
}
sealed class BoolTensor(native: pytorch.Tensor) extends Tensor[Bool](native) { /* 10 Bool */
  requireNativeType(ScalarType.Bool)
  override def dtype: Bool = bool
}
sealed class QInt8Tensor(native: pytorch.Tensor) extends Tensor[QInt8](native) { /* 11 */
  requireNativeType(ScalarType.QInt8)
  override def dtype: QInt8 = qint8
}
sealed class QUInt8Tensor(native: pytorch.Tensor) extends Tensor[QUInt8](native) { /* 12 */
  requireNativeType(ScalarType.QUInt8)
  override def dtype: QUInt8 = quint8
}
sealed class QInt32Tensor(native: pytorch.Tensor) extends Tensor[QInt32](native) { /* 13 */
  requireNativeType(ScalarType.QInt32)
  override def dtype: QInt32 = qint32
}
sealed class BFloat16Tensor(native: pytorch.Tensor) extends Tensor[BFloat16](native) { /* 14 */
  requireNativeType(ScalarType.BFloat16)
  override def dtype: BFloat16 = bfloat16
}
sealed class QUInt4x2Tensor(native: pytorch.Tensor) extends Tensor[Complex32](native) { /* 15 */
  requireNativeType(ScalarType.QUInt4x2)
  override def dtype: Complex32 = complex32
}
sealed class QUInt2x4Tensor(native: pytorch.Tensor) extends Tensor[QUInt4x2](native) { /* 16 */
  requireNativeType(ScalarType.QUInt2x4)
  override def dtype: QUInt4x2 = quint4x2
}
sealed class Float16Tensor(native: pytorch.Tensor) extends Tensor[Float16](native) { /* 17, Half */
  requireNativeType(ScalarType.Half)
  override def dtype: Float16 = float16
}
sealed class UndefinedTensor(native: pytorch.Tensor) extends Tensor[Undefined](native) { /* 18 */
  requireNativeType(ScalarType.Undefined)
  override def dtype: Undefined = undefined
}
sealed class NumOptionsTensor(native: pytorch.Tensor) extends Tensor[NumOptions](native) { /* 19 */
  requireNativeType(ScalarType.NumOptions)
  override def dtype: NumOptions = numoptions
}

type IntTensor = Int8Tensor | UInt8Tensor | Int16Tensor | Int32Tensor | Int64Tensor
type ComplexTensor = Complex32Tensor | Complex64Tensor | Complex128Tensor

object Tensor:
  def apply[D <: DType](native: pytorch.Tensor): Tensor[D] = (native.scalar_type().intern() match
    case ScalarType.Byte          => new Int8Tensor(native)
    case ScalarType.Char          => new UInt8Tensor(native)
    case ScalarType.Short         => new Int16Tensor(native)
    case ScalarType.Int           => new Int32Tensor(native)
    case ScalarType.Long          => new Int64Tensor(native)
    case ScalarType.Half          => new Float16Tensor(native)
    case ScalarType.Float         => new Float32Tensor(native)
    case ScalarType.Double        => new Float64Tensor(native)
    case ScalarType.ComplexHalf   => new Complex32Tensor(native)
    case ScalarType.ComplexFloat  => new Complex64Tensor(native)
    case ScalarType.ComplexDouble => new Complex128Tensor(native)
    case ScalarType.Bool          => new BoolTensor(native)
    case ScalarType.QInt8         => new QInt8Tensor(native)
    case ScalarType.QUInt8        => new QUInt8Tensor(native)
    case ScalarType.QInt32        => new QInt32Tensor(native)
    case ScalarType.BFloat16      => new BFloat16Tensor(native)
    case ScalarType.QUInt4x2      => new QUInt4x2Tensor(native)
    case ScalarType.QUInt2x4      => new QUInt2x4Tensor(native)
    case ScalarType.Undefined     => new UndefinedTensor(native)
    case ScalarType.NumOptions    => new NumOptionsTensor(native)
  ).asInstanceOf[Tensor[D]]

  /** Constructs a tensor with no autograd history (also known as a “leaf tensor”) by copying data.
    */
  // TODO support multidimensional arrays as input
  // TODO support explicit dtype
  def apply[U <: ScalaType: ClassTag](
      data: Seq[U] | U,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[ScalaToDType[U]] =
    data match
      case data: Seq[?] =>
        val (pointer, inputDType) = data.toArray match
          case bools: Array[Boolean] =>
            (
              {
                val p = new BoolPointer(bools.length)
                for ((b, i) <- bools.zipWithIndex) p.put(i, b)
                p
              },
              bool
            )
          case bytes: Array[Byte]     => (new BytePointer(ByteBuffer.wrap(bytes)), int8)
          case shorts: Array[Short]   => (new ShortPointer(ShortBuffer.wrap(shorts)), int16)
          case ints: Array[Int]       => (new IntPointer(IntBuffer.wrap(ints)), int32)
          case longs: Array[Long]     => (new LongPointer(LongBuffer.wrap(longs)), int64)
          case floats: Array[Float]   => (new FloatPointer(FloatBuffer.wrap(floats)), float32)
          case doubles: Array[Double] => (new DoublePointer(DoubleBuffer.wrap(doubles)), float64)
          case _                      => throw new IllegalArgumentException("Unsupported type")
        Tensor(
          torchNative
            .from_blob(
              pointer,
              Array(data.length.toLong),
              NativeConverters.tensorOptions(inputDType, layout, device, requiresGrad)
            )
        )
      case data: U =>
        val dtype = scalaToDType(data)
        Tensor(
          torchNative.scalar_tensor(
            NativeConverters.toScalar(data),
            NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
          )
        )
      case _ => throw new IllegalArgumentException("Unsupported type")
