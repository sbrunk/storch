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
