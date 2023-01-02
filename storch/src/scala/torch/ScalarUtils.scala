package torch

import org.bytedeco.pytorch
import org.bytedeco.pytorch.Scalar
import org.bytedeco.pytorch.global.torch.ScalarType
import spire.math.Complex

private[torch] object ScalarUtils:
  def toScalar(x: ScalaType): pytorch.Scalar = x match
    case x: Boolean => pytorch.Scalar(if true then 1: Byte else 0: Byte)
    case x: Byte    => pytorch.Scalar(x)
    case x: Short   => pytorch.Scalar(x)
    case x: Int     => pytorch.Scalar(x)
    case x: Long    => pytorch.Scalar(x)
    case x: Float   => pytorch.Scalar(x)
    case x: Double  => pytorch.Scalar(x)
    case x @ Complex(r: Float, i: Float) => ???
    case x @ Complex(r: Double, i: Double) => ???


//  def complexFloatToScala(s: Scalar): Complex[Float] =
//    val b = s.asByteBuffer().asFloatBuffer()
//    val r = b.get(0)
//    val i = b.get(1)
//    Complex(r,i)
