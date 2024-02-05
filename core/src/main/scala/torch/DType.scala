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

import org.bytedeco.pytorch.global.torch.ScalarType

import java.nio.{Buffer, ByteBuffer, DoubleBuffer, FloatBuffer, IntBuffer, LongBuffer, ShortBuffer}
import spire.math.{Complex, UByte}

import scala.compiletime.{erasedValue, summonFrom}

// format: off
/** A [[torch.DType]] is an object that represents the data type of a [[torch.Tensor]]. PyTorch has twelve different
  * data types:
  *
  * | Data type                | dtype                                 |
  * |--------------------------|---------------------------------------|
  * | 32-bit floating point    | `torch.float32` or `torch.float`      |
  * | 64-bit floating point    | `torch.float64` or `torch.double`     |
  * | 64-bit complex           | `torch.complex64` or `torch.cfloat`   |
  * | 128-bit complex          | `torch.complex128` or `torch.cdouble` |
  * | 16-bit floating point[1] | `torch.float16` or `torch.half`       |
  * | 16-bit floating point[2] | `torch.bfloat16`                      |
  * | 8-bit integer (unsigned) | `torch.uint8`                         |
  * | 8-bit integer (signed)   | `torch.int8`                          |
  * | 16-bit integer (signed)  | `torch.int16` or `torch.short`        |
  * | 32-bit integer (signed)  | `torch.int32` or `torch.int`          |
  * | 64-bit integer (signed)  | `torch.int64` or `torch.long`         |
  * | Boolean                  | `torch.bool`                          |
  * 
  * To find out if a `torch.dtype` is a floating point data type, the
  * property `is_floating_point` can be used, which returns `True` if the
  * data type is a floating point data type.
  * 
  * To find out if a `torch.dtype` is a complex data type, the property
  * `is_complex` can be used, which returns `True` if the data type is a
  * complex data type.
  * 
  * When the dtypes of inputs to an arithmetic operation (<span
  * class="title-ref">add</span>, <span class="title-ref">sub</span>, <span
  * class="title-ref">div</span>, <span class="title-ref">mul</span>)
  * differ, we promote by finding the minimum dtype that satisfies the
  * following rules:
  * 
  * -   If the type of a scalar operand is of a higher category than tensor
  *     operands (where complex \> floating \> integral \> boolean), we
  *     promote to a type with sufficient size to hold all scalar operands
  *     of that category.
  * -   If a zero-dimension tensor operand has a higher category than
  *     dimensioned operands, we promote to a type with sufficient size and
  *     category to hold all zero-dim tensor operands of that category.
  * -   If there are no higher-category zero-dim operands, we promote to a
  *     type with sufficient size and category to hold all dimensioned
  *     operands.
  * 
  * A floating point scalar operand has dtype <span
  * class="title-ref">torch.get_default_dtype()</span> and an integral
  * non-boolean scalar operand has dtype <span
  * class="title-ref">torch.int64</span>. Unlike numpy, we do not inspect
  * values when determining the minimum <span
  * class="title-ref">dtypes</span> of an operand. Quantized and complex
  * types are not yet supported.
  * 
  * [1] Sometimes referred to as binary16: uses 1 sign, 5 exponent, and 10
  * significand bits. Useful when precision is important.
  * 
  * [2] Sometimes referred to as Brain Floating Point: use 1 sign, 8
  * exponent and 7 significand bits. Useful when range is important, since
  * it has the same number of exponent bits as `float32`
  */
// format: on
sealed abstract class DType private[torch] ():
  private[torch] def toScalarType: ScalarType = this match
    case _: UInt8         => ScalarType.Byte
    case _: Int8          => ScalarType.Char
    case _: Int16         => ScalarType.Short
    case _: Int32         => ScalarType.Int
    case _: Int64         => ScalarType.Long
    case _: Float16       => ScalarType.Half
    case _: Float32       => ScalarType.Float
    case _: Float64       => ScalarType.Double
    case _: Complex32     => ScalarType.ComplexHalf
    case _: Complex64     => ScalarType.ComplexFloat
    case _: Complex128    => ScalarType.ComplexDouble
    case _: Bool          => ScalarType.Bool
    case _: QInt8         => ScalarType.QInt8
    case _: QUInt8        => ScalarType.QUInt8
    case _: QInt32        => ScalarType.QInt32
    case _: BFloat16      => ScalarType.BFloat16
    case _: QUInt4x2      => ScalarType.QUInt4x2
    case _: QUInt2x4      => ScalarType.QUInt2x4
    case _: Bits1x8       => ScalarType.Bits1x8
    case _: Bits2x4       => ScalarType.Bits2x4
    case _: Bits4x2       => ScalarType.Bits4x2
    case _: Bits8         => ScalarType.Bits8
    case _: Bits16        => ScalarType.Bits16
    case _: Float8_e5m2   => ScalarType.Float8_e5m2
    case _: Float8_e4m3fn => ScalarType.Float8_e4m3fn
    case _: Undefined     => ScalarType.Undefined
    case _: NumOptions    => ScalarType.NumOptions

object DType:
  private[torch] def fromScalarType(t: ScalarType): DType = t.intern() match
    case ScalarType.Byte          => uint8
    case ScalarType.Char          => int8
    case ScalarType.Short         => int16
    case ScalarType.Int           => int32
    case ScalarType.Long          => int64
    case ScalarType.Float         => float32
    case ScalarType.Double        => float64
    case ScalarType.ComplexHalf   => complex32
    case ScalarType.ComplexFloat  => complex64
    case ScalarType.ComplexDouble => complex128
    case ScalarType.Bool          => bool
    case ScalarType.QInt8         => qint8
    case ScalarType.QUInt8        => quint8
    case ScalarType.QInt32        => qint32
    case ScalarType.BFloat16      => bfloat16
    case ScalarType.QUInt4x2      => quint4x2
    case ScalarType.QUInt2x4      => quint2x4
    case ScalarType.Bits1x8       => bits1x8
    case ScalarType.Bits2x4       => bits2x4
    case ScalarType.Bits4x2       => bits4x2
    case ScalarType.Bits8         => bits8
    case ScalarType.Bits16        => bits16
    case ScalarType.Float8_e5m2   => float8_e5m2
    case ScalarType.Float8_e4m3fn => float8_e4m3fn
    case ScalarType.Half          => float16
    case ScalarType.Undefined     => undefined
    case ScalarType.NumOptions    => numoptions
  case object uint8 extends UInt8 /* 0, */
  case object int8 extends Int8 /* 1, */
  case object int16 extends Int16 /* 2, */
  case object int32 extends Int32 /* 3, */
  case object int64 extends Int64 /* 4, */
  case object float16 extends Float16 /* 5*/
  case object float32 extends Float32 /* 6 */
  case object float64 extends Float64 /* 7 */
  case object complex32 extends Complex32 /* 8 */
  case object complex64 extends Complex64 /* 9 */
  case object complex128 extends Complex128 /* 10 */
  case object bool extends Bool /* 11 */
  case object qint8 extends QInt8 /* 12 */
  case object quint8 extends QUInt8 /* 13 */
  case object qint32 extends QInt32 /* 14 */
  case object bfloat16 extends BFloat16 /* 15 */
  case object quint4x2 extends QUInt4x2 /* 16 */
  case object quint2x4 extends QUInt2x4 /* 17 */
  case object bits1x8 extends Bits1x8 /* 18 */
  case object bits2x4 extends Bits2x4 /* 19 */
  case object bits4x2 extends Bits4x2 /* 20 */
  case object bits8 extends Bits8 /* 21 */
  case object bits16 extends Bits16 /* 22 */
  case object float8_e5m2 extends Float8_e5m2 /* 23 */
  case object float8_e4m3fn extends Float8_e4m3fn /* 24 */
  case object undefined extends Undefined /* 25 */
  case object numoptions extends NumOptions /* 26 */

sealed abstract class UInt8 extends DType /* 0, Byte */
sealed abstract class Int8 extends DType /* 1, Char */
sealed abstract class Int16 extends DType /* 2, Short */
sealed abstract class Int32 extends DType /* 3, Int */
sealed abstract class Int64 extends DType /* 4, Long */
sealed abstract class Float16 extends DType /* 5, Half */
sealed abstract class Float32 extends DType /* 6, Float */
sealed abstract class Float64 extends DType /* 7, Double */
sealed abstract class Complex32 extends DType /* 8, ComplexHalf */
sealed abstract class Complex64 extends DType /* 9, ComplexFloat */
sealed abstract class Complex128 extends DType /* 10, ComplexDouble */
sealed abstract class Bool extends DType /* 10 */
sealed abstract class QInt8 extends DType /* 11 */
sealed abstract class QUInt8 extends DType /* 12 */
sealed abstract class QInt32 extends DType /* 13 */
sealed abstract class BFloat16 extends DType /* 14 */
sealed abstract class QUInt4x2 extends DType /* 15 */
sealed abstract class QUInt2x4 extends DType /* 16 */
sealed abstract class Bits1x8 extends DType
sealed abstract class Bits2x4 extends DType
sealed abstract class Bits4x2 extends DType
sealed abstract class Bits8 extends DType
sealed abstract class Bits16 extends DType
sealed abstract class Float8_e5m2 extends DType
sealed abstract class Float8_e4m3fn extends DType
sealed abstract class Undefined extends DType
sealed abstract class NumOptions extends DType

val uint8: UInt8 = DType.uint8
val int8: Int8 = DType.int8
val int16: Int16 = DType.int16
val int32: Int32 = DType.int32
val int64: Int64 = DType.int64
val float16: Float16 = DType.float16
val float32: Float32 = DType.float32
val float64: Float64 = DType.float64
val complex32: Complex32 = DType.complex32
val complex64: Complex64 = DType.complex64
val complex128: Complex128 = DType.complex128
val bool: Bool = DType.bool
val qint8: QInt8 = DType.qint8
val quint8: QUInt8 = DType.quint8
val qint32: QInt32 = DType.qint32
val bfloat16: BFloat16 = DType.bfloat16
val quint4x2: QUInt4x2 = DType.quint4x2
val quint2x4: QUInt2x4 = DType.quint2x4
val bits1x8: Bits1x8 = DType.bits1x8
val bits2x4: Bits2x4 = DType.bits2x4
val bits4x2: Bits4x2 = DType.bits4x2
val bits8: Bits8 = DType.bits8
val bits16: Bits16 = DType.bits16
val float8_e5m2: Float8_e5m2 = DType.float8_e5m2
val float8_e4m3fn: Float8_e4m3fn = DType.float8_e4m3fn
val undefined: Undefined = DType.undefined
val numoptions: NumOptions = DType.numoptions

sealed class Derive private ()
private object Derive:
  val derive: Derive = Derive()
export Derive.derive

/** Default tensor type.
  *
  * Defaults to float32 but can be overriden by providing providing the DType explicitly, or by
  * overriding the default in the current scope through an import:
  *
  * Example:
  * ```scala sc
  * import torch.*
  *
  * // Default:
  * nn.Linear(10, 10) // Linear[Float32]
  *
  * // Override explicitly:
  * nn.Linear[BFloat16](10, 10) // Linear[BFloat16]
  *
  * // Override default:
  * import Default.float64
  * nn.Linear(10, 10) // Linear[Float64]
  * ```
  */
trait Default[+D <: DType]:
  def dtype: D
trait LowPriorityDefaults:
  given float16: Default[Float16] = new Default[Float16] { def dtype = torch.float16 }
  given bfloat16: Default[BFloat16] = new Default[BFloat16] { def dtype = torch.bfloat16 }
  given float64: Default[Float64] = new Default[Float64] { def dtype = torch.float64 }
  given complex32: Default[Complex32] = new Default[Complex32] { def dtype = torch.complex32 }
  given complex64: Default[Complex64] = new Default[Complex64] { def dtype = torch.complex64 }
  given complex128: Default[Complex128] = new Default[Complex128] { def dtype = torch.complex128 }
object Default extends LowPriorityDefaults:
  given float32: Default[Float32] = new Default[Float32] { def dtype = torch.float32 }

/** DType combinations * */
type FloatNN = Float16 | Float32 | Float64 | BFloat16

type IntNN = Int8 | UInt8 | Int16 | Int32 | Int64

type ComplexNN = Complex32 | Complex64 | Complex128

type BitwiseNN = Bool | IntNN

type NumericRealNN = IntNN | FloatNN

type RealNN = NumericRealNN | Bool

type NumericNN = NumericRealNN | ComplexNN

/** Scala type combinations * */
type NumericReal = Byte | UByte | Short | Int | Long | Float | Double

type Real = NumericReal | Boolean

type ComplexScala = Complex[Float] | Complex[Double]

type Numeric = NumericReal | ComplexScala

type ScalaType = Real | ComplexScala

type DTypeToScala[T <: DType] <: ScalaType = T match
  case UInt8      => UByte
  case Int8       => Byte
  case Int16      => Short
  case Int32      => Int
  case Int64      => Long
  case Float16    => Float
  case Float32    => Float
  case Float64    => Double
  case Complex32  => Complex[Float]
  case Complex64  => Complex[Float]
  case Complex128 => Complex[Double]
  case Bool       => Boolean
  case _          => ScalaType
  // TODO remaining types

type ScalaToDType[S <: ScalaType] <: DType = S match
  case UByte           => UInt8
  case Byte            => Int8
  case Short           => Int16
  case Int             => Int32
  case Long            => Int64
  case Float           => Float32
  case Double          => Float64
  case Boolean         => Bool
  case Complex[Float]  => Complex64
  case Complex[Double] => Complex128

def scalaToDType[S <: ScalaType](s: S): DType = s match
  case _: UByte                      => uint8
  case _: Byte                       => int8
  case _: Short                      => int16
  case _: Int                        => int32
  case _: Long                       => int64
  case _: Float                      => float32
  case _: Double                     => float64
  case _: Boolean                    => bool
  case Complex(_: Double, _: Double) => complex128
  case Complex(_: Float, _: Float)   => complex64

type TensorType[T] <: DType = T match
  case UInt8         => UInt8
  case Int8          => Int8
  case Int16         => Int16
  case Int32         => Int32
  case Int64         => Int64
  case Float16       => Float16
  case Float32       => Float32
  case Float64       => Float64
  case Complex32     => Complex32
  case Complex64     => Complex64
  case Complex128    => Complex128
  case Bool          => Bool
  case QInt8         => QInt8
  case QUInt8        => QUInt8
  case QInt32        => QInt32
  case BFloat16      => BFloat16
  case QUInt4x2      => QUInt4x2
  case QUInt2x4      => QUInt2x4
  case Bits1x8       => Bits1x8
  case Bits2x4       => Bits2x4
  case Bits4x2       => Bits4x2
  case Bits8         => Bits8
  case Bits16        => Bits16
  case Float8_e5m2   => Float8_e5m2
  case Float8_e4m3fn => Float8_e4m3fn
  case Undefined     => Undefined
  case NumOptions    => NumOptions
  case DType         => DType

type DTypeOrDeriveFromScalar[T <: DType | Derive, U <: ScalaType] <: DType = T match
  case Derive => ScalaToDType[U]
  case T      => TensorType[T]

type DTypeOrDeriveFromTensor[D1 <: DType, U <: DType | Derive] <: DType = U match
  case Derive => D1
  case U      => TensorType[U]

type PromotedDType[A <: DType, B <: DType] <: Float32 | Int32 | Int64 = (A, B) match
  case (Float64, B) => Float32
  case (A, Float64) => Float32
  case (Float32, B) => Float32
  case (A, Float32) => Float32
  case (Int64, B)   => Int64
  case (A, Int64)   => Int64
  case (Int32, B)   => Int32
  case (A, Int32)   => Int32
  case _            => Int64

private[torch] def promotedDType[A <: DType, B <: DType](a: A, b: B) = (a, b) match
  case (_: Float64, _) | (_, _: Float64) => float32
  case (_: Float32, _) | (_, _: Float32) => float32
  case (_: Int64, _) | (_, _: Int64)     => int64
  case (_: Int32, _) | (_, _: Int32)     => int32
  case _                                 => int64

type DerivedArangeType[Start <: ScalaType, End <: ScalaType, Step <: ScalaType] =
  PromotedDType[ScalaToDType[Start], PromotedDType[ScalaToDType[End], ScalaToDType[Step]]]

private[torch] def derivedArangeType[Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
    start: Start,
    end: End,
    step: Step
): DType = promotedDType(scalaToDType(start), promotedDType(scalaToDType(end), scalaToDType(step)))

type DTypeOrDeriveArange[
    T <: DType | Derive,
    Start <: ScalaType,
    End <: ScalaType,
    Step <: ScalaType
] <: DType =
  T match
    case Derive => DerivedArangeType[Start, End, Step]
    case T      => TensorType[T]

/** Type of the output tensor based on PyTorch type promotion rules
  *
  * This is a type-level implementation of the PyTorch op data type promotion rules via match types.
  *
  * @see
  *   [Reference implementation of PyTorch datatype promotion
  *   rules](https://github.com/pytorch/pytorch/blob/fb6749d977e33b5f463c2d0a1b56a939428105e5/c10/core/ScalarType.h#L423-L444)
  */
type Promoted[T <: DType, U <: DType] <: DType = (T, U) match
  case (T, T)                                    => T
  case (U, U)                                    => U
  case (Undefined, U) | (T, Undefined)           => Undefined
  case (Bool, U)                                 => U
  case (T, Bool)                                 => T
  case (Int8, UInt8) | (UInt8, Int8)             => Int16
  case (UInt8, U)                                => U
  case (T, UInt8)                                => T
  case (Int8, U)                                 => U
  case (T, Int8)                                 => T
  case (Int16, U)                                => U
  case (T, Int16)                                => T
  case (Int32, U)                                => U
  case (T, Int32)                                => T
  case (Int64, U)                                => U
  case (T, Int64)                                => T
  case (Float8_e5m2, U)                          => U
  case (T, Float8_e5m2)                          => T
  case (Float8_e4m3fn, U)                        => U
  case (T, Float8_e5m2)                          => T
  case (Float16, BFloat16) | (BFloat16, Float16) => Float32
  case (Float16, U)                              => U
  case (T, Float16)                              => T
  case (Float32, U)                              => U
  case (T, Float32)                              => T
  case (Float64, U)                              => U
  case (T, Float64)                              => T
  case (Complex32, U)                            => U
  case (T, Complex32)                            => T
  case (Complex64, U)                            => U
  case (T, Complex64)                            => T
  case (Complex128, U)                           => U
  case (T, Complex128)                           => T
  case _                                         => DType

/** Promoted type for tensor operations that always output numbers (e.g. `square`) */
type NumericPromoted[D <: DType] <: DType = D match
  case Bool => Int64
  case _    => D

/** Promoted type for tensor operations that always output floats (e.g. `sin`) */
type FloatPromoted[D <: DType] <: FloatNN | ComplexNN = D match
  case Float16    => Float16
  case BFloat16   => BFloat16
  case Float64    => Float64
  case Complex32  => Complex32
  case Complex64  => Complex64
  case Complex128 => Complex128
  case _          => Float32

/** Demoted type for complex to real type extractions (e.g. `imag`, `real`) */
type ComplexToReal[D <: DType] <: DType = D match
  case Complex32  => Float16
  case Complex64  => Float32
  case Complex128 => Float64
  case _          => D

/** Promoted type for tensor operations that always output full sized complex or real (e.g.
  * `floatPower`)
  */
type ComplexPromoted[T <: DType, U <: DType] <: Float64 | Complex128 = (T, U) match
  case (ComplexNN, U) => Complex128
  case (T, ComplexNN) => Complex128
  case _              => Float64

/** Promoted type for tensor division */
type Div[T <: DType, U <: DType] <: DType = (T, U) match
  case (BitwiseNN, BitwiseNN) => Float32
  case _                      => Promoted[T, U]

/** Promoted type for elementwise tensor sum */
type Sum[D <: DType] <: DType = D match
  case BitwiseNN => Int64
  case D         => D

private[torch] type TypedBuffer[T <: ScalaType] <: Buffer = T match
  case Short  => ShortBuffer
  case Int    => IntBuffer
  case Long   => LongBuffer
  case Float  => FloatBuffer
  case Double => DoubleBuffer
  // case Complex[Float]   => FloatBuffer
  // case Complex[Double]  => DoubleBuffer
  case Byte => ByteBuffer

transparent inline def deriveDType[T <: DType]: DType =
  inline erasedValue[T] match
    case _: UInt8         => uint8
    case _: Int8          => int8
    case _: Int16         => int16
    case _: Int32         => int32
    case _: Int64         => int64
    case _: Float16       => float16
    case _: Float32       => float32
    case _: Float64       => float64
    case _: Complex32     => complex32
    case _: Complex64     => complex64
    case _: Complex128    => complex128
    case _: Bool          => bool
    case _: QInt8         => qint8
    case _: QUInt8        => quint8
    case _: QInt32        => qint32
    case _: BFloat16      => bfloat16
    case _: QUInt4x2      => quint4x2
    case _: QUInt2x4      => quint2x4
    case _: Bits1x8       => bits1x8
    case _: Bits2x4       => bits2x4
    case _: Bits4x2       => bits4x2
    case _: Bits8         => bits8
    case _: Bits16        => bits16
    case _: Float8_e5m2   => float8_e5m2
    case _: Float8_e4m3fn => float8_e4m3fn
    case _: Undefined     => undefined
    case _: NumOptions    => numoptions
