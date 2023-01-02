package torch

case class Slice(start: Option[Long], end: Option[Long], step: Option[Long])
object Slice:
  private def extract(index: Option[Long] | Long) = index match
    case i: Option[Long] => i
    case i: Long         => Option(i)
  def apply(
    start: Option[Long] | Long = None,
    end: Option[Long] | Long = None,
    step: Option[Long] | Long = None
  ): Slice = Slice(extract(start), extract(end), extract(step))