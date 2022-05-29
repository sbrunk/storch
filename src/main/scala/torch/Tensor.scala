package torch

import org.bytedeco.javacpp.indexer.Indexer
import org.bytedeco.pytorch

import scala.annotation.targetName


class Tensor(private val underlying: pytorch.Tensor):
  @targetName("mul")
  def *(u: Tensor):Tensor = Tensor(underlying.mul(u.underlying))
  @targetName("mul")
  def *(s: Int): Tensor = Tensor(underlying.mul(pytorch.Scalar(s)))

  override def toString: String = underlying.toString + underlying.createIndexer[Indexer]().toString


object Tensor:
  private[torch] def apply(underlying: pytorch.Tensor): Tensor = new Tensor(underlying)

  @targetName("fromIntSeq")
  def apply(x: Seq[Int], shape: Long*): Tensor = Tensor(pytorch.AbstractTensor.create(x.toArray, shape *))
  @targetName("fromDoubleSeq")
  def apply(x: Seq[Double], shape: Long*): Tensor = Tensor(pytorch.AbstractTensor.create(x.toArray, shape*))
  @targetName("fromByte")
  def apply(x: Byte*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))
  @targetName("fromShort")
  def apply(x: Short*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))
  @targetName("fromInt")
  def apply(x: Int*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))
  @targetName("fromLong")
  def apply(x: Long*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))
  @targetName("fromFloat")
  def apply(x: Float*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))
  @targetName("fromDouble")
  def apply(x: Double*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))
  @targetName("fromBoolean")
  def apply(x: Boolean*): Tensor = Tensor(pytorch.AbstractTensor.create(x *))