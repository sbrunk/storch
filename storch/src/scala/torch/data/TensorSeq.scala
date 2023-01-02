package torch
package data

import scala.util.Random

/** Wraps a tensor as a Seq.
  *
  * Each sample will be retrieved by indexing tensors along the first dimension.
  *
  * @param t
  *   tensor to be wrapped as a seq
  */
class TensorSeq[D <: DType](t: Tensor[D]) extends IndexedSeq[Tensor[D]] {

  require(t.size.length > 0)
  require(t.size.head <= Int.MaxValue)

  override def apply(i: Int): Tensor[D] = t(i)

  override def length: Int = t.size.head.toInt

}