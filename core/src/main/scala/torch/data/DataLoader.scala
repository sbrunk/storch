package torch
package data

import scala.util.Random

/** Provides an iterable over batches of a given dataset. */
class DataLoader[Input, Batch](
    dataset: IndexedSeq[Input],
    batchSize: Int = 1,
    shuffle: Boolean = false,
    collateFn: Seq[Input] => Batch
) extends Iterable[Batch] {

  override def iterator =
    (if shuffle then Random.shuffle(dataset) else dataset)
      .grouped(batchSize)
      .map(collateFn)
}

class TupleDataLoader[D1 <: DType, D2 <: DType](
    dataset: IndexedSeq[(Tensor[D1], Tensor[D2])],
    batchSize: Int = 1,
    shuffle: Boolean = false,
    collateFn: Seq[(Tensor[D1], Tensor[D2])] => (Tensor[D1], Tensor[D2]) = (examples: Seq[(Tensor[D1], Tensor[D2])]) =>
      (torch.stack(examples.map(_._1)), torch.stack(examples.map(_._2)))
) extends DataLoader[(Tensor[D1], Tensor[D2]), (Tensor[D1], Tensor[D2])](dataset, batchSize, shuffle, collateFn)

class ExampleDataLoader[D1 <: DType, D2 <: DType](
    dataset: IndexedSeq[Example[D1, D2]],
    batchSize: Int = 1,
    shuffle: Boolean = false,
    collateFn: Seq[Example[D1, D2]] => (Tensor[D1], Tensor[D2]) = (examples: Seq[Example[D1, D2]]) =>
      (torch.stack(examples.map(_.feature)), torch.stack(examples.map(_.target)))
) extends DataLoader[Example[D1, D2], (Tensor[D1], Tensor[D2])](dataset, batchSize, shuffle, collateFn)
