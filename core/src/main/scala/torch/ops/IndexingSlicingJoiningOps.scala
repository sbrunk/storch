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
package ops

import org.bytedeco.pytorch.{TensorArrayRef, TensorVector}
import org.bytedeco.pytorch.global.torch as torchNative

/** Indexing, Slicing, Joining, Mutating Ops
  *
  * https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
  */
private[torch] trait IndexingSlicingJoiningOps {

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
// TODO slice_scatterd
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
}
