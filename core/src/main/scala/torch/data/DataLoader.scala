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
    collateFn: Seq[(Tensor[D1], Tensor[D2])] => (Tensor[D1], Tensor[D2]) =
      (examples: Seq[(Tensor[D1], Tensor[D2])]) =>
        (torch.stack(examples.map(_._1)), torch.stack(examples.map(_._2)))
) extends DataLoader[(Tensor[D1], Tensor[D2]), (Tensor[D1], Tensor[D2])](
      dataset,
      batchSize,
      shuffle,
      collateFn
    )

class ExampleDataLoader[D1 <: DType, D2 <: DType](
    dataset: IndexedSeq[Example[D1, D2]],
    batchSize: Int = 1,
    shuffle: Boolean = false,
    collateFn: Seq[Example[D1, D2]] => (Tensor[D1], Tensor[D2]) =
      (examples: Seq[Example[D1, D2]]) =>
        (torch.stack(examples.map(_.feature)), torch.stack(examples.map(_.target)))
) extends DataLoader[Example[D1, D2], (Tensor[D1], Tensor[D2])](
      dataset,
      batchSize,
      shuffle,
      collateFn
    )
