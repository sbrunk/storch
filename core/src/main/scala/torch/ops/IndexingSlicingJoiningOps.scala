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

import internal.NativeConverters.*

import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.TensorArrayRef
import org.bytedeco.pytorch.TensorVector

/** Indexing, Slicing, Joining, Mutating Ops
  *
  * https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
  */
private[torch] trait IndexingSlicingJoiningOps {

  private def toArrayRef(tensors: Seq[Tensor[?]]): TensorArrayRef =
    new TensorArrayRef(new TensorVector(tensors.map(_.native)*))

  /** Returns a view of the tensor conjugated and with the last two dimensions transposed.
    *
    * `x.adjoint()` is equivalent to `x.transpose(-2, -1).conj()` for complex tensors and to
    * `x.transpose(-2, -1)` for real tensors.
    *
    * Example:
    *
    * ```scala sc
    * import spire.math.Complex
    *
    * val x = torch.arange(end = 4)
    * // tensor dtype=int32, shape=[4], device=CPU
    * // [0, 1, 2, 3]
    * val a = torch.Tensor(
    *   Seq(
    *     Seq(Complex(0.0, 0.0), Complex(1.0, 1.0)),
    *     Seq(Complex(2.0, 2.0), Complex(3.0, 3.0))
    *   )
    * )
    * // tensor dtype=complex128, shape=[2, 2], device=CPU
    * // [[(0.0 + 0.0i), (1.0 + 1.0i)],
    * //  [(2.0 + 2.0i), (3.0 + 3.0i)]]
    * a.adjoint()
    * // tensor dtype=complex128, shape=[2, 2], device=CPU
    * // [[(0.0 - 0.0i), (2.0 - 2.0i)],
    * //  [(1.0 - 1.0i), (3.0 - 3.0i)]]
    * (a.adjoint() == a.mH).all()
    * // tensor dtype=bool, shape=[], device=CPU
    * // true
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def adjoint[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    torchNative.adjoint(input.native)
  )

  /** Returns a tensor containing the indices of all non-zero elements of `input`. Each row in the
    * result contains the indices of a non-zero element in `input`. The result is sorted
    * lexicographically, with the last index changing the fastest (C-style).
    *
    * If `input` has *n* dimensions, then the resulting indices tensor `out` is of size (*z*×*n*),
    * where *z* is the total number of non-zero elements in the `input` tensor.
    *
    * Note When `input` is on CUDA, this function causes host-device synchronization.
    *
    * Example:
    * ```scala sc
    * val t = torch.Tensor(Seq(1, 0, 1))
    * torch.argwhere(t)
    * // tensor dtype=int32, shape=[2, 1], device=CPU
    * // [[0],
    * //  [2]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def argwhere[D <: DType](input: Tensor[D]): Tensor[Int64] = fromNative(
    torchNative.argwhere(input.native)
  )

  /** Concatenates the given sequence of `seq` tensors in the given dimension. All tensors must
    * either have the same shape (except in the concatenating dimension) or be empty.
    *
    * `torch.cat` can be seen as an inverse operation for `torch.split` and `torch.chunk`.
    *
    * `torch.cat`{class: interpreted-text, role: func} can be best understood via examples.
    *
    * Non-empty tensors provided must have the same shape, except in the cat dimension.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.randn(Seq(2, 3))
    * // tensor dtype=float32, shape=[2, 3], device=CPU
    * // [[ 0.6580, -1.0969, -0.4614],
    * //  [-0.1034, -0.5790,  0.1497]])
    * torch.cat(Seq(x, x, x), 0)
    * // tensor dtype=float32, shape=[6, 3], device=CPU
    * // [[ 0.6580, -1.0969, -0.4614],
    * //  [-0.1034, -0.5790,  0.1497],
    * //  [ 0.6580, -1.0969, -0.4614],
    * //  [-0.1034, -0.5790,  0.1497],
    * //  [ 0.6580, -1.0969, -0.4614],
    * //  [-0.1034, -0.5790,  0.1497]])
    * torch.cat(Seq(x, x, x), 1)
    * // tensor dtype=float32, shape=[2, 9], device=CPU
    * // [[ 0.6580, -1.0969, -0.4614, ..., 0.6580, -1.0969, -0.4614],
    * //  [-0.1034, -0.5790,  0.1497, ..., -0.1034, -0.5790,  0.1497]])
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def cat[D <: DType](tensors: Seq[Tensor[D]], dim: Int = 0): Tensor[D] = fromNative(
    torchNative.cat(toArrayRef(tensors), dim.toLong)
  )

  /** Returns a view of `input` with a flipped conjugate bit. If `input` has a non-complex dtype,
    * this function just returns `input`.
    *
    * Note
    *
    * `torch.conj` performs a lazy conjugation, but the actual conjugated tensor can be materialized
    * at any time using `torch.resolve_conj`.
    *
    * Warning
    *
    * In the future, `torch.conj` may return a non-writeable view for an `input` of non-complex
    * dtype. It's recommended that programs not modify the tensor returned by `torch.conj_physical`
    * when `input` is of non-complex dtype to be compatible with this change.
    *
    * Example:
    *
    * ```scala sc
    * import spire.math.Complex
    * val x = torch.Tensor(
    *   Seq(Complex(-1.0, 1.0), Complex(-2.0, 2.0), Complex(3.0, 3.0))
    * )
    * // tensor dtype=complex128, shape=[3], device=CPU
    * // [(-1.0 + 1.0i), (-2.0 + 2.0i), (3.0 + 3.0i)]
    * x.isConj
    * // false
    * val y = torch.conj(x)
    * // tensor dtype=complex128, shape=[3], device=CPU
    * // [(-1.0 + 1.0i), (-2.0 + 2.0i), (3.0 + 3.0i)]
    * y.isConj
    * // true
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def conj[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.conj(input.native))

  /** Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the
    * input tensor.
    *
    * This function may return fewer than the specified number of chunks!
    *
    * `torch.tensorSplit` a function that always returns exactly the specified number of chunks
    *
    * If the tensor size along the given dimension `dim` is divisible by `chunks`, all returned
    * chunks will be the same size. If the tensor size along the given dimension `dim` is not
    * divisible by `chunks`, all returned chunks will be the same size, except the last one. If such
    * division is not possible, this function may return fewer than the specified number of chunks.
    *
    * Example:
    *
    * ```scala sc
    * torch.chunk(torch.arange(end = 11), 6)
    * // List(
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [0, 1],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [2, 3],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [4, 5],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [6, 7],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [8, 9],
    * //   tensor dtype=int32, shape=[1], device=CPU
    * //   [10]
    * // )
    * torch.chunk(torch.arange(end = 12), 6)
    * // List(
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [0, 1],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [2, 3],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [4, 5],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [6, 7],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [8, 9],
    * //   tensor dtype=int32, shape=[2], device=CPU
    * //   [10, 11]
    * // )
    * torch.chunk(torch.arange(end = 13), 6)
    * // List(
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [0, 1, 2],
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [3, 4, 5],
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [6, 7, 8],
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [9, 10, 11],
    * //   tensor dtype=int32, shape=[1], device=CPU
    * //   [12],
    * // )
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def chunk[D <: DType](input: Tensor[D], chunks: Int, dim: Int = 0): Seq[Tensor[D]] = {
    val tensors = torchNative.chunk(input.native, chunks, dim.toLong)
    (0L until tensors.size()).map(i => fromNative(tensors.get(i)))
  }

  /** Splits `input`, a tensor with three or more dimensions, into multiple tensors depthwise
    * according to `indices_or_sections`. Each split is a view of `input`.
    *
    * This is equivalent to calling torch.tensorSplit(input, indicesOrSections, dim=2) (the split
    * dimension is 2), except that if `indicesOrSections` is an integer it must evenly divide the
    * split dimension or a runtime error will be thrown.
    *
    * Example:
    *
    * ```scala sc
    * // val t = torch.arange(end = 16.0).reshape(2, 2, 4)
    * // tensor dtype=float32, shape=[2, 2, 4], device=CPU
    * // [[[0.0, 1.0, 2.0, 3.0],
    * //   [4.0, 5.0, 6.0, 7.0]],
    * //  [[8.0, 9.0, 10.0, 11.0],
    * //   [12.0, 13.0, 14.0, 15.0]]]
    * // torch.dsplit(t, 2)
    * // List(
    * //   tensor dtype=float32, shape=[2, 2, 2], device=CPU
    * //   [[[0.0, 1.0],
    * //     [4.0, 5.0]],
    * //    [[8.0, 9.0],
    * //     [12.0, 13.0]]],
    * //   tensor dtype=float32, shape=[2, 2, 2], device=CPU
    * //   [[[2.0, 3.0],
    * //     [6.0, 7.0]],
    * //    [[10.0, 11.0],
    * //     [14.0, 15.0]]]
    * // )
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def dsplit[D <: DType](input: Tensor[D], indicesOrSections: Int*): Seq[Tensor[D]] = {
    val tensors = torchNative.dsplit(input.native, indicesOrSections.map(_.toLong)*)
    (0L until tensors.size()).map(i => fromNative(tensors.get(i)))
  }

  /** Creates a new tensor by horizontally stacking the tensors in `tensors`.
    *
    * Equivalent to `torch.hstack(tensors)`, except each zero or one dimensional tensor `t` in
    * `tensors` is first reshaped into a `(t.numel(), 1)` column before being stacked horizontally.
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.Tensor(Seq(1, 2, 3))
    * val b = torch.Tensor(Seq(4, 5, 6))
    * torch.columnStack(Seq(a, b))
    * // tensor dtype=int32, shape=[3, 2], device=CPU
    * // [[1, 4],
    * //  [2, 5],
    * //  [3, 6]]
    * val c = torch.arange(end = 5)
    * val d = torch.arange(end = 10).reshape(5, 2)
    * torch.columnStack(Seq(c, d, d))
    * // tensor dtype=int32, shape=[5, 5], device=CPU
    * // [[0, 0, 1, 0, 1],
    * //  [1, 2, 3, 2, 3],
    * //  [2, 4, 5, 4, 5],
    * //  [3, 6, 7, 6, 7],
    * //  [4, 8, 9, 8, 9]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def columnStack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] =
    fromNative(torchNative.column_stack(toArrayRef(tensors)))

  /** Stack tensors in sequence depthwise (along third axis).
    *
    * This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been
    * reshaped by `torch.atleast3d`
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.Tensor(Seq(1, 2, 3))
    * val b = torch.Tensor(Seq(4, 5, 6))
    * torch.dstack(Seq(a, b))
    * // tensor dtype=int32, shape=[1, 3, 2], device=CPU
    * // [[[1, 4],
    * //   [2, 5],
    * //   [3, 6]]]
    * val c = torch.Tensor(Seq(Seq(1), Seq(2), Seq(3)))
    * val d = torch.Tensor(Seq(Seq(4), Seq(5), Seq(6)))
    * torch.dstack(Seq(c, d))
    * // tensor dtype=int32, shape=[3, 1, 2], device=CPU
    * // [[[1, 4]],
    * //  [[2, 5]],
    * //  [[3, 6]]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def dstack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] = fromNative(
    torchNative.dstack(toArrayRef(tensors))
  )

  /** Gathers values along an axis specified by `dim`.
    *
    * For a 3-D tensor the output is specified by:
    *
    * ```
    * out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    * out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    * out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    * ```
    *
    * `input` and `index` must have the same number of dimensions. It is also required that
    * `index.size(d) <= input.size(d)` for all dimensions `d != dim`. `out` will have the same shape
    * as `index`. Note that `input` and `index` do not broadcast against each other.
    *
    * Example:
    *
    * ```scala sc
    * val t = torch.Tensor(Seq(Seq(1, 2), Seq(3, 4)))
    * val index = torch.Tensor(Seq(Seq(0L, 0L), Seq(1L, 0L)))
    * torch.gather(t, 1, index)
    * // tensor dtype=int32, shape=[2, 2], device=CPU
    * // [[1, 1],
    * //  [4, 3]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dim
    *   axis along which to index
    * @param index
    *   indices of elements to gather
    * @param sparseGrad
    *   if `true`, gradient w.r.t. `input` will be a sparse tensor
    */
  def gather[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      sparseGrad: Boolean = false
  ): Tensor[D] =
    fromNative(torchNative.gather(input.native, dim.toLong, index.native, sparseGrad))

  /** Splits `input`, a tensor with one or more dimensions, into multiple tensors horizontally
    * according to `indices_or_sections`. Each split is a view of `input`.
    *
    * If `input` is one dimensional this is equivalent to calling `torch.tensorSplit(input,
    * indicesOrSections, dim=0)` (the split dimension is zero), and if `input` has two or more
    * dimensions it's equivalent to calling torch.tensorSplit(input, indicesOrSections, dim=1) (the
    * split dimension is 1), except that if `indicesOrSections` is an integer it must evenly divide
    * the split dimension or a runtime error will be thrown.
    *
    * Example:
    *
    * ```scala sc
    * val t = torch.arange(end = 16.0).reshape(4, 4)
    * torch.hsplit(t, 2)
    * // List(
    * //   tensor dtype=float32, shape=[4, 2], device=CPU
    * //   [[0.0, 1.0],
    * //    [4.0, 5.0],
    * //    [8.0, 9.0],
    * //    [12.0, 13.0]],
    * //   tensor dtype=float32, shape=[4, 2], device=CPU
    * //   [[2.0, 3.0],
    * //    [6.0, 7.0],
    * //    [10.0, 11.0],
    * //    [14.0, 15.0]]
    * // )
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def hsplit[D <: DType](input: Tensor[D], indicesOrSections: Int*): Seq[Tensor[D]] = {
    val tensors = torchNative.hsplit(input.native, indicesOrSections.toArray.map(_.toLong)*)
    (0L until tensors.size()).map(i => fromNative(tensors.get(i)))
  }

  /** Stack tensors in sequence horizontally (column wise).
    *
    * This is equivalent to concatenation along the first axis for 1-D tensors, and along the second
    * axis for all other tensors.
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.Tensor(Seq(1, 2, 3))
    * val b = torch.Tensor(Seq(4, 5, 6))
    * torch.hstack(Seq(a, b))
    * // tensor dtype=int32, shape=[6], device=CPU
    * // [1, 2, 3, 4, 5, 6]
    * val c = torch.Tensor(Seq(Seq(1), Seq(2), Seq(3)))
    * val d = torch.Tensor(Seq(Seq(4), Seq(5), Seq(6)))
    * torch.hstack(Seq(c, d))
    * // tensor dtype=int32, shape=[3, 2], device=CPU
    * // [[1, 4],
    * //  [2, 5],
    * //  [3, 6]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def hstack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] =
    fromNative(torchNative.hstack(toArrayRef(tensors)))

  /** Accumulate the elements of `source` into the `input` tensor by adding to the indices in the
    * order given in `index`.
    *
    * The dimth dimension of source must have the same size as the length of index (which must be a
    * vector), and all other dimensions must match self, or an error will be raised.
    *
    * Note:
    *
    * This operation may behave nondeterministically when given tensors on a CUDA device. See
    * Reproducibility for more information.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.ones(Seq(5, 3))
    * val index = torch.Tensor(Seq(0L, 4L, 2L))
    * val t = torch.Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9))).to(dtype = torch.float32)
    * torch.indexAdd(x, 0, index, t)
    * // tensor dtype=float32, shape=[5, 3], device=CPU
    * // [[2.0, 3.0, 4.0],
    * //  [1.0, 1.0, 1.0],
    * //  [8.0, 9.0, 10.0],
    * //  [1.0, 1.0, 1.0],
    * //  [5.0, 6.0, 7.0]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def indexAdd[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      source: Tensor[D]
  ): Tensor[D] =
    fromNative(torchNative.index_add(input.native, dim.toLong, index.native, source.native))

  /** Copies the elements of tensor into the self tensor by selecting the indices in the order given
    * in index. For example, if dim == 0 and index[i] == j, then the ith row of tensor is copied to
    * the jth row of self.
    *
    * The dimth dimension of tensor must have the same size as the length of index (which must be a
    * vector), and all other dimensions must match self, or an error will be raised.
    *
    * Note:
    *
    * If index contains duplicate entries, multiple elements from tensor will be copied to the same
    * index of self. The result is nondeterministic since it depends on which copy occurs last.
    *
    * Example:
    * ```scala sc
    * val x = torch.zeros(Seq(5, 3))
    * val t = torch.Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9))).to(dtype = torch.float32)
    * // tensor dtype=float32, shape=[3, 3], device=CPU
    * // [[1.0, 2.0, 3.0],
    * //  [4.0, 5.0, 6.0],
    * //  [7.0, 8.0, 9.0]]
    * val index = torch.Tensor(Seq(0L, 4L, 2L))
    * torch.indexCopy(x, 0, index, t)
    * // tensor dtype=float32, shape=[5, 3], device=CPU
    * // [[1.0, 2.0, 3.0],
    * //  [0.0, 0.0, 0.0],
    * //  [7.0, 8.0, 9.0],
    * //  [0.0, 0.0, 0.0],
    * //  [4.0, 5.0, 6.0]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def indexCopy[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      source: Tensor[D]
  ): Tensor[D] =
    fromNative(torchNative.index_copy(input.native, dim.toLong, index.native, source.native))

  // TODO index_reduce
  // TODO Enum for reduce: String
  // def indexReduce[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64], reduce: String): Tensor[D] =
  //   Tensor(torchNative.index_reduce(input.native, dim.toLong, index.native))

  /** Returns a new tensor which indexes the `input` tensor along dimension `dim` using the entries
    * in `index` which is a `Tensor[Int64]`.
    *
    * The returned tensor has the same number of dimensions as the original tensor (`input`). The
    * `dim`th dimension has the same size as the length of `index`; other dimensions have the same
    * size as in the original tensor.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.randn(Seq(3, 4))
    * // tensor dtype=float32, shape=[3, 4], device=CPU
    * // [[ 0.1427,  0.0231, -0.5414, -1.0009],
    * //  [-0.4664,  0.2647, -0.1228, -1.1068],
    * //  [-1.1734, -0.6571,  0.7230, -0.6004]]
    * val indices = torch.Tensor(Seq(0L, 2L))
    * torch.indexSelect(x, 0, indices)
    * // tensor dtype=float32, shape=[2, 4], device=CPU
    * // [[ 0.1427,  0.0231, -0.5414, -1.0009],
    * //  [-1.1734, -0.6571,  0.7230, -0.6004]]
    * torch.indexSelect(x, 1, indices)
    * // tensor dtype=float32, shape=[2, 4], device=CPU
    * // [[ 0.1427, -0.5414],
    * //  [-0.4664, -0.1228],
    * //  [-1.1734,  0.7230]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def indexSelect[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64]): Tensor[D] =
    fromNative(torchNative.index_select(input.native, dim.toLong, index.native))

  /** Returns a new 1-D tensor which indexes the `input` tensor according to the boolean mask `mask`
    * which is a <span class="title-ref">BoolTensor</span>.
    *
    * The shapes of the `mask` tensor and the `input` tensor don't need to match, but they must be
    * `broadcastable <broadcasting-semantics>`.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.randn(Seq(3, 4))
    * // tensor dtype=float32, shape=[3, 4], device=CPU
    * // [[ 0.3552, -2.3825, -0.8297,  0.3477],
    * //  [-1.2035,  1.2252,  0.5002,  0.6248],
    * //  [ 0.1307, -2.0608,  0.1244,  2.0139]]
    *
    * val mask = x.ge(0.5)
    * // tensor dtype=bool, shape=[3, 4], device=CPU
    * // [[false, false, false, false],
    * //  [false, true, true, true],
    * //  [false, false, false, true]]
    *
    * torch.maskedSelect(x, mask)
    * // tensor dtype=float32, shape=[4], device=CPU
    * // [1.2252, 0.5002, 0.6248, 2.0139]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def maskedSelect[D <: DType](input: Tensor[D], mask: Tensor[Bool]): Tensor[D] =
    fromNative(torchNative.masked_select(input.native, mask.native))

  /** Moves the dimension(s) of `input` at the position(s) in `source` to the position(s) in
    * `destination`.
    *
    * Other dimensions of `input` that are not explicitly moved remain in their original order and
    * appear at the positions not specified in `destination`.
    *
    * Examples:
    *
    * ```scala sc
    * val t = torch.randn(Seq(3, 2, 1))
    * // tensor dtype=float32, shape=[3, 2, 1], device=CPU
    * // [[[-0.3362],
    * //   [-0.8437]],
    * //  [[-0.9627],
    * //   [ 0.1727]],
    * //  [[ 0.5173],
    * //   [-0.1398]]]
    *
    * torch.movedim(t, 1, 0).shape
    * // Seq(2, 3, 1)
    *
    * torch.movedim(t, 1, 0)
    * // tensor dtype=float32, shape=[2, 3, 1], device=CPU
    * // [[[-0.3362],
    * //   [-0.9627],
    * //   [ 0.5173]],
    * //  [[-0.8437],
    * //   [ 0.1727],
    * //   [-0.1398]]]
    *
    * torch.movedim(t, Seq(1, 2), Seq(0, 1)).shape
    * // Seq(2, 1, 3)
    *
    * torch.movedim(t, Seq(1, 2), Seq(0, 1))
    * // tensor dtype=float32, shape=[2, 1, 3], device=CPU
    * // [[[-0.3362, -0.9627,  0.5173]],
    * //  [[-0.8437,  0.1727, -0.1398]]]
    * ```
    *
    * @param source
    *   Original positions of the dims to move. These must be unique destination
    * @param destination
    *   Destination positions for each of the original dims. These must also be unique.
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def movedim[D <: DType](input: Tensor[D], source: Int, destination: Int): Tensor[D] =
    fromNative(torchNative.movedim(input.native, source.toLong, destination.toLong))
  def movedim[D <: DType](input: Tensor[D], source: Seq[Int], destination: Seq[Int]): Tensor[D] =
    fromNative(
      torchNative.movedim(input.native, source.map(_.toLong).toArray, destination.map(_.toLong)*)
    )

  /** Alias for `torch.movedim`.
    *
    * Examples:
    *
    * ```scala sc
    * val t = torch.randn(Seq(3, 2, 1))
    * // tensor dtype=float32, shape=[3, 2, 1], device=CPU
    * // [[[-0.3362],
    * //   [-0.8437]],
    * //  [[-0.9627],
    * //   [ 0.1727]],
    * //  [[ 0.5173],
    * //   [-0.1398]]]
    *
    * torch.moveaxis(t, 1, 0).shape
    * // Seq(2, 3, 1)
    *
    * torch.moveaxis(t, 1, 0)
    * // tensor dtype=float32, shape=[2, 3, 1], device=CPU
    * // [[[-0.3362],
    * //   [-0.9627],
    * //   [ 0.5173]],
    * //  [[-0.8437],
    * //   [ 0.1727],
    * //   [-0.1398]]]
    *
    * torch.moveaxis(t, Seq(1, 2), Seq(0, 1)).shape
    * // Seq(2, 1, 3)
    *
    * torch.moveaxis(t, Seq(1, 2), Seq(0, 1))
    * // tensor dtype=float32, shape=[2, 1, 3], device=CPU
    * // [[[-0.3362, -0.9627,  0.5173]],
    * //  [[-0.8437,  0.1727, -0.1398]]]
    * ```
    *
    * @param source
    *   * Original positions of the dims to move. These must be unique destination
    * @param destination
    *   Destination positions for each of the original dims. These must also be unique.
    *
    * @group indexing_slicing_joining_mutating_ops
    */

  def moveaxis[D <: DType](input: Tensor[D], source: Int, destination: Int): Tensor[D] =
    fromNative(torchNative.moveaxis(input.native, source.toLong, destination.toLong))
  def moveaxis[D <: DType](input: Tensor[D], source: Seq[Int], destination: Seq[Int]): Tensor[D] =
    fromNative(
      torchNative.moveaxis(input.native, source.map(_.toLong).toArray, destination.map(_.toLong)*)
    )

    /** Returns a new tensor that is a narrowed version of `input` tensor. The dimension `dim` is
      * input from `start` to `start + length`. The returned tensor and `input` tensor share the
      * same underlying storage.
      *
      * Args: input (Tensor): the tensor to narrow dim (int): the dimension along which to narrow
      * start (int or Tensor): index of the element to start the narrowed dimension from. Can be
      * negative, which means indexing from the end of <span class="title-ref">dim</span>. If <span
      * class="title-ref">Tensor</span>, it must be an 0-dim integral <span
      * class="title-ref">Tensor</span> (bools not allowed) length (int): length of the narrowed
      * dimension, must be weakly positive
      *
      * Example:
      *
      * ```scala sc
      * val x = torch.arange(start = 1, end = 10).reshape(3, 3)
      * // tensor dtype=int32, shape=[3, 3], device=CPU
      * // [[ 1,  2,  3],
      * //  [ 4,  5,  6],
      * //  [ 7,  8,  9]]
      *
      * torch.narrow(x, 0, 0, 2)
      * // tensor dtype=int32, shape=[2, 3], device=CPU
      * // [[ 1,  2,  3],
      * //  [ 4,  5,  6]]
      *
      * torch.narrow(x, 1, 1, 2)
      * // tensor dtype=int32, shape=[3, 2], device=CPU
      * // [[ 2,  3],
      * //  [ 5,  6],
      * //  [ 8,  9]]
      *
      * torch.narrow(x, -1, -1, 1)
      * // tensor dtype=int32, shape=[3, 1], device=CPU
      * // [[3],
      * //  [6],
      * //  [9]]
      * ```
      *
      * @param dim
      *   the dimension along which to narrow
      * @param start
      *   index of the element to start the narrowed dimension from. Can be negative, which means
      *   indexing from the end of `dim`
      * @param length
      *   length of the narrowed dimension, must be weakly positive
      *
      * @group indexing_slicing_joining_mutating_ops
      */
  def narrow[D <: DType](input: Tensor[D], dim: Int, start: Int, length: Int): Tensor[D] =
    fromNative(torchNative.narrow(input.native, dim.toLong, start.toLong, length.toLong))

  /** Same as `torch.narrow` except this returns a copy rather than shared storage. This is
    * primarily for sparse tensors, which do not have a shared-storage narrow method.
    *
    * Keyword args: {out}
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.arange(start = 1, end = 10).reshape(3, 3)
    * // tensor dtype=int32, shape=[3, 3], device=CPU
    * // [[ 1,  2,  3],
    * //  [ 4,  5,  6],
    * //  [ 7,  8,  9]]
    *
    * torch.narrowCopy(x, 0, 0, 2)
    * // tensor dtype=int32, shape=[2, 3], device=CPU
    * // [[ 1,  2,  3],
    * //  [ 4,  5,  6]]
    *
    * torch.narrowCopy(x, 1, 1, 2)
    * // tensor dtype=int32, shape=[3, 2], device=CPU
    * // [[ 2,  3],
    * //  [ 5,  6],
    * //  [ 8,  9]]
    *
    * torch.narrowCopy(x, -1, -1, 1)
    * // tensor dtype=int32, shape=[3, 1], device=CPU
    * // [[3],
    * //  [6],
    * //  [9]]
    * ```
    *
    * @param dim
    *   the dimension along which to narrow
    * @param start
    *   index of the element to start the narrowed dimension from. Can be negative, which means
    *   indexing from the end of `dim`
    * @param length
    *   length of the narrowed dimension, must be weakly positive
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def narrowCopy[D <: DType](input: Tensor[D], dim: Int, start: Int, length: Int): Tensor[D] =
    fromNative(torchNative.narrow_copy(input.native, dim.toLong, start.toLong, length.toLong))

  /** Returns a tensor containing the indices of all non-zero elements of `input`. Each row in the
    * result contains the indices of a non-zero element in `input`. The result is sorted
    * lexicographically, with the last index changing the fastest (C-style).
    *
    * If `input` has *n* dimensions, then the resulting indices tensor `out` is of size (*z*×*n*),
    * where *z* is the total number of non-zero elements in the `input` tensor.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.Tensor(Seq(1, 1, 1, 0, 1))
    * torch.nonzero(x)
    * // tensor dtype=int32, shape=[4, 1], device=CPU
    * // [[ 0],
    * //  [ 1],
    * //  [ 2],
    * //  [ 4]]
    *
    * val y = torch.Tensor(
    *   Seq(
    *     Seq(0.6, 0.0, 0.0, 0.0),
    *     Seq(0.0, 0.4, 0.0, 0.0),
    *     Seq(0.0, 0.0, 1.2, 0.0),
    *     Seq(0.0, 0.0, 0.0, -0.4)
    *   )
    * )
    * torch.nonzero(y)
    * // tensor dtype=int32, shape=[4, 2], device=CPU
    * // [[ 0,  0],
    * //  [ 1,  1],
    * //  [ 2,  2],
    * //  [ 3,  3]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def nonzero[D <: DType](input: Tensor[D]): Tensor[Int64] = fromNative(
    torchNative.nonzero(input.native)
  )

  /** Returns a view of the original tensor `input` with its dimensions permuted.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.randn(Seq(2, 3, 5))
    * // tensor dtype=float32, shape=[2, 3, 5], device=CPU
    * // ...
    *
    * torch.permute(x, 2, 0, 1)
    * // tensor dtype=float32, shape=[5, 2, 3], device=CPU
    * // ...
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dims
    *   The desired ordering of dimensions, must be unique
    */
  def permute[D <: DType](input: Tensor[D], dims: Int*): Tensor[D] =
    fromNative(torchNative.permute(input.native, dims.map(_.toLong)*))

  /** Returns a tensor with the same data and number of elements as `input`, but with the specified
    * shape. When possible, the returned tensor will be a view of `input`. Otherwise, it will be a
    * copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying,
    * but you should not depend on the copying vs. viewing behavior.
    *
    * See [[torch.Tensor.view]] on when it is possible to return a view.
    *
    * A single dimension may be -1, in which case it's inferred from the remaining dimensions and
    * the number of elements in `input`.
    *
    * Args: input (Tensor): the tensor to be reshaped shape (tuple of int): the new shape
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.arange(end = 4.0)
    * torch.reshape(a, 2, 2)
    * // tensor dtype=float32, shape=[2, 2], device=CPU
    * // [[0.0, 1.0],
    * //  [2.0, 3.0]]
    *
    * val b = torch.Tensor(Seq(Seq(0, 1), Seq(2, 3)))
    * torch.reshape(b, -1)
    * // tensor dtype=int32, shape=[4], device=CPU
    * // [0, 1, 2, 3]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def reshape[D <: DType](input: Tensor[D], shape: Int*): Tensor[D] =
    fromNative(torchNative.reshape(input.native, shape.map(_.toLong)*))

  /** Slices the `input` tensor along the selected dimension at the given index. This function
    * returns a view of the original tensor with the given dimension removed.
    *
    * Note
    *
    * `select` is equivalent to slicing. For example:
    *
    * `torch.select(tensor, 0, index)` is equivalent to `tensor(index)`
    *
    * `torch.select(tensor, 2, index)` is equivalent to `tensor(---, ---,index)`.
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dim
    *   the dimension to slice
    * @param index
    *   the index to select with
    */
  def select[D <: DType](input: Tensor[D], dim: Int, index: Int): Tensor[D] =
    fromNative(torchNative.select(input.native, dim.toLong, index.toLong))

  // TODO Add docs for scatter
  // TODO Add reduction arg
  def scatter[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      source: Tensor[D]
  ): Tensor[D] =
    fromNative(torchNative.scatter(input.native, dim.toLong, index.native, source.native))

  // TODO Add docs diagonalScatter
  def diagonalScatter[D <: DType](
      input: Tensor[D],
      src: Tensor[D],
      offset: Int = 0,
      dim1: Int = 0,
      dim2: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.diagonal_scatter(
        input.native,
        src.native,
        offset.toLong,
        dim1.toLong,
        dim2.toLong
      )
    )

  // TODO Add docs selectScatter
  def selectScatter[D <: DType](input: Tensor[D], src: Tensor[D], dim: Int, index: Int): Tensor[D] =
    fromNative(torchNative.select_scatter(input.native, src.native, dim.toLong, index.toLong))

  // TODO Add docs for sliceScatterd
  // TODO Review default start and end
  def sliceScatter[D <: DType](
      input: Tensor[D],
      src: Tensor[D],
      dim: Int,
      start: Int | Option[Int] = None,
      end: Int | Option[Int] = None,
      step: Int = 1
  ): Tensor[D] =
    fromNative(
      torchNative.slice_scatter(
        input.native,
        src.native,
        dim.toLong,
        start.toOptional,
        end.toOptional,
        step.toLong
      )
    )

  // TODO Add docs for scatter_add
  def scatterAdd[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      src: Tensor[D]
  ): Tensor[D] =
    fromNative(torchNative.scatter_add(input.native, dim.toLong, index.native, src.native))

  // TODO scatter_reduce
  // TODO enum for reduce options?
  // def scatterReduce[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64], src: Tensor[D], reduce: String, includeSelf: Boolean): Tensor[D] =
  //   fromNative(torchNative.scatter_reduce(input.native, dim.toLong, index.native, src.native, reduce))

  /** Splits the tensor into chunks. Each chunk is a view of the original tensor.
    *
    * If `splitSizeOrSections` is an integer type, then tensor will be split into equally sized
    * chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension
    * dim is not divisible by `splitSize`.
    *
    * If `splitSizeOrSections` is a list, then tensor will be split into len(`splitSizeOrSections`)
    * chunks with sizes in dim according to `splitSizeOrSections`.
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.arange(end = 10).reshape(5, 2)
    * // tensor dtype=int32, shape=[5, 2], device=CPU
    * // [[0, 1],
    * //  [2, 3],
    * //  [4, 5],
    * //  [6, 7],
    * //  [8, 9]]
    *
    * torch.split(a, 2)
    * // List(
    * //   tensor dtype=int32, shape=[2, 2], device=CPU
    * //   [[0, 1],
    * //    [2, 3]],
    * //   tensor dtype=int32, shape=[2, 2], device=CPU
    * //   [[4, 5],
    * //    [6, 7]],
    * //   tensor dtype=int32, shape=[1, 2], device=CPU
    * //   [[8, 9]]
    * // )
    *
    * torch.split(a, Seq(1, 4))
    * // List(
    * //   tensor dtype=int32, shape=[1, 2], device=CPU
    * //   [[0, 1]]
    * //   tensor dtype=int32, shape=[4, 2], device=CPU
    * //   [[2, 3],
    * //    [4, 5],
    * //    [6, 7],
    * //    [8, 9]]
    * // )
    * ```
    *
    * @param input
    * @param splitSizeOrSections
    * @param dim
    */
  def split[D <: DType](
      input: Tensor[D],
      splitSizeOrSections: Int | Seq[Int],
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result =
      splitSizeOrSections match {
        case i: Int      => torchNative.split(input.native, i.toLong, dim.toLong)
        case s: Seq[Int] => torchNative.split(input.native, s.map(_.toLong).toArray, dim.toLong)
      }
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  /** Returns a tensor with all specified dimensions of `input` of size 1 removed.
    *
    * For example, if `input` is of shape: (*A*×1×*B*×*C*×1×*D*) then the `input.squeeze()` will be
    * of shape: (*A*×*B*×*C*×*D*).
    *
    * When `dim` is given, a squeeze operation is done only in the given dimension(s). If `input` is
    * of shape: (*A*×1×*B*), `squeeze(input, 0)` leaves the tensor unchanged, but `squeeze(input,
    * 1)` will squeeze the tensor to the shape (*A*×*B*).
    *
    * Note
    *
    * The returned tensor shares the storage with the input tensor, so changing the contents of one
    * will change the contents of the other.
    *
    * Warning
    *
    * If the tensor has a batch dimension of size 1, then `squeeze(input)` will also remove the
    * batch dimension, which can lead to unexpected errors. Consider specifying only the dims you
    * wish to be squeezed.
    *
    * Example::
    *
    * ```scala sc
    * val x = torch.zeros(Seq(2, 1, 2, 1, 2))
    * x.size
    * // Seq(2, 1, 2, 1, 2)
    * val a = torch.squeeze(x)
    * a.size
    * // Seq(2, 2, 2)
    * val b = torch.squeeze(x, 0)
    * b.size
    * // Seq(2, 1, 2, 1, 2)
    * val c = torch.squeeze(x, 1)
    * c.size
    * // Seq(2, 2, 1, 2)
    * val d = torch.squeeze(x, 1, 2, 3)
    * d.size
    * // Seq(2, 2, 2)
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dim
    *   if given, the input will be squeezed only in the specified dimensions.
    */
  def squeeze[D <: DType](input: Tensor[D], dim: Int*): Tensor[D] =
    // NOTE: There are different runtime behaviours betwen passing an empty dim and passing no dim, so we need to check for it
    dim match {
      case Nil => fromNative(torchNative.squeeze(input.native))
      case _   => fromNative(torchNative.squeeze(input.native, dim.map(_.toLong)*))
    }

  /** Concatenates a sequence of tensors along a new dimension.
    *
    * All tensors need to be of the same size.
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dim
    *   dimension to insert. Has to be between 0 and the number of dimensions of concatenated
    *   tensors (inclusive)
    */
  def stack[D <: DType](tensors: Seq[Tensor[D]], dim: Int = 0): Tensor[D] = fromNative(
    torchNative.stack(toArrayRef(tensors), dim)
  )

  /** Alias for `torch.transpose`.
    *
    * This function is equivalent to NumPy's swapaxes function.
    *
    * Examples::
    *
    * ```scala sc
    * val x = torch.Tensor(Seq(Seq(Seq(0, 1), Seq(2, 3)), Seq(Seq(4, 5), Seq(6, 7))))
    * // tensor dtype=int32, shape=[2, 2, 2], device=CPU
    * // [[[0, 1],
    * //   [2, 3]],
    * //
    * //  [[4, 5],
    * //   [6, 7]]]
    *
    * torch.swapaxes(x, 0, 1)
    * // tensor dtype=int32, shape=[2, 2, 2], device=CPU
    * // [[[0, 1],
    * //   [4, 5]],
    * //
    * //  [[2, 3],
    * //   [6, 7]]]
    *
    * torch.swapaxes(x, 0, 2)
    * // tensor dtype=int32, shape=[2, 2, 2], device=CPU
    * // [[[0, 4],
    * //   [2, 6]],
    * //
    * //  [[1, 5],
    * //   [3, 7]]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def swapaxes[D <: DType](input: Tensor[D], axis1: Int, axis2: Int): Tensor[D] =
    fromNative(torchNative.swapaxes(input.native, axis1.toLong, axis2.toLong))

  /** Alias for `torch.transpose`.
    *
    * This function is equivalent to NumPy's swapaxes function.
    *
    * Examples:
    *
    * ```scala sc
    * val x = torch.Tensor(Seq(Seq(Seq(0, 1), Seq(2, 3)), Seq(Seq(4, 5), Seq(6, 7))))
    * // tensor dtype=int32, shape=[2, 2, 2], device=CPU
    * // [[[0, 1],
    * //   [2, 3]],
    * //
    * //  [[4, 5],
    * //   [6, 7]]]
    *
    * torch.swapdims(x, 0, 1)
    * // tensor dtype=int32, shape=[2, 2, 2], device=CPU
    * // [[[0, 1],
    * //   [4, 5]],
    * //
    * //  [[2, 3],
    * //   [6, 7]]]
    *
    * torch.swapdims(x, 0, 2)
    * // tensor dtype=int32, shape=[2, 2, 2], device=CPU
    * // [[[0, 4],
    * //   [2, 6]],
    * //
    * //  [[1, 5],
    * //   [3, 7]]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def swapdims[D <: DType](input: Tensor[D], axis1: Int, axis2: Int): Tensor[D] =
    fromNative(torchNative.swapdims(input.native, axis1.toLong, axis2.toLong))

  /** Expects `input` to be <= 2-D tensor and transposes dimensions 0 and 1.
    *
    * 0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to
    * `transpose(input, 0, 1)`.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.arange(end = 6).reshape(2, 3)
    * // tensor dtype=int32, shape=[2, 3], device=CPU
    * // [[0, 1, 2],
    * //  [3, 4, 5]]
    * torch.t(x)
    * // tensor dtype=int32, shape=[3, 2], device=CPU
    * // [[0, 3],
    * //  [1, 4],
    * //  [2, 5]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def t[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.t(input.native))

  /** Returns a new tensor with the elements of `input` at the given indices. The input tensor is
    * treated as if it were viewed as a 1-D tensor. The result takes the same shape as the indices.
    *
    * Example:
    *
    * ```scala sc
    * val src = torch.Tensor(Seq(Seq(4, 3, 5), Seq(6, 7, 8)))
    * val index = torch.Tensor(Seq(0L, 2L, 5L))
    * torch.take(src, index)
    * // tensor dtype=int32, shape=[3], device=CPU
    * // [4, 5, 8]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def take[D <: DType](input: Tensor[D], index: Tensor[Int64]): Tensor[D] =
    fromNative(torchNative.take(input.native, index.native))

  /** Selects values from `input` at the 1-dimensional indices from `indices` along the given `dim`.
    *
    * Functions that return indices along a dimension, like `torch.argmax` and `torch.argsort`, are
    * designed to work with this function. See the examples below.
    *
    * Example:
    *
    * ```scala sc
    * val t = torch.Tensor(Seq(Seq(10, 30, 20), Seq(60, 40, 50)))
    * val maxIdx = torch.argmax(t)
    * torch.takeAlongDim(t, maxIdx)
    * // tensor dtype=int32, shape=[1], device=CPU
    * // [60]
    * val sortedIdx = torch.argsort(t, dim = 1)
    * torch.takeAlongDim(t, sortedIdx, dim = 1)
    * // tensor([[10, 20, 30],
    * //         [40, 50, 60]])
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param index
    *   the indices into `input`
    * @param dim
    *   dimension to select along.
    */
  // TODO index Int | Tensor[Int64]
  def takeAlongDim[D <: DType](
      input: Tensor[D],
      index: Tensor[Int64],
      dim: Int | Option[Int] = None
  ): Tensor[D] =
    fromNative(torchNative.take_along_dim(input.native, index.native, dim.toOptional))

  /** Splits a tensor into multiple sub-tensors, all of which are views of `input`, along dimension
    * `dim` according to the indices or number of sections specified by `indices_or_sections`. This
    * function is based on NumPy's `numpy.array_split`.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.arange(end = 8)
    * torch.tensorSplit(x, 3)
    * // List(
    * //   [0, 1, 2],
    * //   [3, 4, 5],
    * //   [6, 7]
    * // )
    *
    * val y = torch.arange(end = 7)
    * torch.tensorSplit(y, 3)
    * // List(
    * //   [0, 1, 2],
    * //   [3, 4],
    * //   [5, 6]
    * // )
    *
    * torch.tensorSplit(x, Seq(1, 6))
    * // List(
    * //   [0],
    * //   [1, 2, 3, 4, 5],
    * //   [6]
    * // )
    *
    * val z = torch.arange(end = 14).reshape(2, 7)
    * torch.tensorSplit(z, 3, dim = 1)
    * // List(
    * //   [[0, 1, 2],
    * //    [7, 8, 9]],
    * //   [[ 3,  4],
    * //    [10, 11]],
    * //   [[ 5,  6],
    * //    [12, 13]]
    * // )
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param input
    *   the tensor to split indices_or_sections (Tensor, int or list or tuple of ints): If
    *   `indices_or_sections` is an integer `n` or a zero dimensional long tensor with value `n`,
    *   `input` is split into `n` sections along dimension `dim`. If `input` is divisible by `n`
    *   along dimension `dim`, each section will be of equal size, `input.size(dim) / n`. If `input`
    *   is not divisible by `n`, the sizes of the first `int(input.size(dim) % n)` sections will
    *   have size `int(input.size(dim) / n) + 1`, and the rest will have size `int(input.size(dim) /
    *   n)`
    * @param splitSizeOrSections
    *   is a list or tuple of ints, or a one-dimensional long > tensor, then `input` is split along
    *   dimension `dim` at each of the indices > in the list, tuple or tensor. For instance,
    *   `indices_or_sections=[2, 3]` and `dim=0` > would result in the tensors `input[:2]`,
    *   `input[2:3]`, and `input[3:]`. > > If `indices_or_sections` is a tensor, it must be a
    *   zero-dimensional or one-dimensional > long tensor on the CPU.
    * @param dim
    *   dimension along which to split the tensor. Default: `0`
    */
  def tensorSplit[D <: DType](
      input: Tensor[D],
      splitSizeOrSections: Int | Seq[Int],
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result = splitSizeOrSections match {
      case i: Int => torchNative.tensor_split(input.native, i.toLong, dim.toLong)
      case s: Seq[Int] =>
        torchNative.tensor_split(input.native, s.toArray.map(_.toLong), dim.toLong)
    }
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  /** Constructs a tensor by repeating the elements of `input`. The `dims` argument specifies the
    * number of repetitions in each dimension.
    *
    * If `dims` specifies fewer dimensions than `input` has, then ones are prepended to `dims` until
    * all dimensions are specified. For example, if `input` has shape (8, 6, 4, 2) and `dims` is (2,
    * 2), then `dims` is treated as (1, 1, 2, 2).
    *
    * Analogously, if `input` has fewer dimensions than `dims` specifies, then `input` is treated as
    * if it were unsqueezed at dimension zero until it has as many dimensions as `dims` specifies.
    * For example, if `input` has shape (4, 2) and `dims` is (3, 3, 2, 2), then `input` is treated
    * as if it had the shape (1, 1, 4, 2).
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.Tensor(Seq(1, 2, 3))
    * torch.tile(x, 2)
    * // tensor dtype=int32, shape=[6], device=CPU
    * // [1, 2, 3, 1, 2, 3]
    * val y = torch.Tensor(Seq(Seq(1, 2), Seq(3, 4)))
    * torch.tile(y, 2, 2)
    * // tensor dtype=int32, shape=[4, 4], device=CPU
    * // [[1, 2, 1, 2],
    * //  [3, 4, 3, 4],
    * //  [1, 2, 1, 2],
    * //  [3, 4, 3, 4]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param reps
    *   the number of repetitions per dimension.
    */
  def tile[D <: DType](input: Tensor[D], reps: Int*): Tensor[D] =
    fromNative(torchNative.tile(input.native, reps.map(_.toLong)*))

  /** Returns a tensor that is a transposed version of `input`. The given dimensions `dim0` and
    * `dim1` are swapped.
    *
    * If `input` is a strided tensor then the resulting `out` tensor shares its underlying storage
    * with the `input` tensor, so changing the content of one would change the content of the other.
    *
    * If `input` is a `sparse tensor` then the resulting `out` tensor *does not* share the
    * underlying storage with the `input` tensor.
    *
    * If `input` is a `sparse tensor` with compressed layout (SparseCSR, SparseBSR, SparseCSC or
    * SparseBSC) the arguments `dim0` and `dim1` must be both batch dimensions, or must both be
    * sparse dimensions. The batch dimensions of a sparse tensor are the dimensions preceding the
    * sparse dimensions.
    *
    * Note
    *
    * Transpositions which interchange the sparse dimensions of a `SparseCSR` or `SparseCSC` layout
    * tensor will result in the layout changing between the two options. Transposition of the sparse
    * dimensions of a ` SparseBSR` or `SparseBSC` layout tensor will likewise generate a result with
    * the opposite layout.
    *
    * Args:
    *
    * : {input} dim0 (int): the first dimension to be transposed dim1 (int): the second dimension to
    * be transposed
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.arange(1, 7).reshape(2, 3)
    * torch.transpose(x, 0, 1)
    * // tensor dtype=int32, shape=[3, 2], device=CPU
    * // [[1, 4],
    * //  [2, 5],
    * //  [3, 6]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def transpose[D <: DType](input: Tensor[D], dim0: Int, dim1: Int): Tensor[D] =
    fromNative(torchNative.transpose(input.native, dim0.toLong, dim1.toLong))

  /** Removes a tensor dimension.
    *
    * Returns a sequence of all slices along a given dimension, already without it.
    *
    * Example:
    *
    * ```scala sc
    * val t = torch.arange(1, 10).reshape(3, 3)
    * torch.unbind(t)
    * // List(
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [1, 2, 3],
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [4, 5, 6],
    * //   tensor dtype=int32, shape=[3], device=CPU
    * //   [7, 8, 9]
    * // )
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dim
    *   dimension to remove
    */
  def unbind[D <: DType](input: Tensor[D], dim: Int = 0): Seq[Tensor[D]] = {
    val result = torchNative.unbind(input.native, dim.toLong)
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  /** Returns a new tensor with a dimension of size one inserted at the specified position.
    *
    * The returned tensor shares the same underlying data with this tensor.
    *
    * A `dim` value within the range `[-input.dim() - 1, input.dim() + 1]` can be used. Negative
    * `dim` will correspond to `unsqueeze` applied at `dim` = `dim + input.dim() + 1`.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.Tensor(Seq(1, 2, 3, 4))
    * torch.unsqueeze(x, 0)
    * // tensor dtype=int32, shape=[1, 4], device=CPU
    * // [[1, 2, 3, 4]]
    * torch.unsqueeze(x, 1)
    * // tensor dtype=int32, shape=[4, 1], device=CPU
    * // [[1],
    * //  [2],
    * //  [3],
    * //  [4]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param dim
    *   the index at which to insert the singleton dimension
    */
  def unsqueeze[D <: DType](input: Tensor[D], dim: Int): Tensor[D] =
    fromNative(torchNative.unsqueeze(input.native, dim.toLong))

  /** Splits `input`, a tensor with two or more dimensions, into multiple tensors vertically
    * according to `indicesOrSections`. Each split is a view of `input`.
    *
    * This is equivalent to calling torch.tensorSplit(input, indicesOrSections, dim=0) (the split
    * dimension is 0), except that if `indicesOrSections` is an integer it must evenly divide the
    * split dimension or a runtime error will be thrown.
    *
    * Example:
    *
    * ```scala sc
    * val t = torch.arange(end = 16.0).reshape(4, 4)
    * torch.vsplit(t, 2)
    * // List(
    * //   tensor dtype=float32, shape=[2, 4], device=CPU
    * //   [[0.0, 1.0, 2.0, 3.0],
    * //    [4.0, 5.0, 6.0, 7.0]],
    * //   tensor dtype=float32, shape=[2, 4], device=CPU
    * //   [[8.0, 9.0, 10.0, 11.0],
    * //    [12.0, 13.0, 14.0, 15.0]]
    * // )
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def vsplit[D <: DType](input: Tensor[D], splitSizeOrSections: Int*): Seq[Tensor[D]] = {
    val result = torchNative.vsplit(input.native, splitSizeOrSections.map(_.toLong)*)
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  /** Stack tensors in sequence vertically (row wise).
    *
    * This is equivalent to concatenation along the first axis after all 1-D tensors have been
    * reshaped by `torch.atleast2d`
    *
    * Example:
    *
    * ```scala sc
    * val a = torch.Tensor(Seq(1, 2, 3))
    * val b = torch.Tensor(Seq(4, 5, 6))
    * torch.vstack(Seq(a, b))
    * // tensor dtype=int32, shape=[2, 3], device=CPU
    * // [[1, 2, 3],
    * //  [4, 5, 6]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    */
  def vstack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] = fromNative(
    torchNative.vstack(toArrayRef(tensors))
  )

  /** Return a tensor of elements selected from either `input` or `other`, depending on `condition`.
    *
    * Note
    *
    * The tensors `condition`, `input`, `other` must be `broadcastable`.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.randn(Seq(3, 2))
    * // tensor dtype=torch.float32, size=[3, 2]
    * // [[-0.4620,  0.3139],
    * //  [ 0.3898, -0.7197],
    * //  [ 0.0478, -0.1657]]
    *
    * torch.where(x > 0, 1.0, 0.0)
    * // tensor dtype=torch.float32, size=[3, 2]
    * // [[0.0, 1.0],
    * //  [1.0, 0.0],
    * //  [1.0, 0.0]]
    *
    * val y = torch.ones(Seq(3, 2))
    * torch.where(x > 0, x, y)
    * // tensor dtype=torch.float32, size=[3, 2]
    * // [[ 1.0,  0.3139],
    * //  [ 0.3898,  1.0],
    * //  [ 0.0478,  1.0]]
    *
    * val z = torch.randn(Seq(2, 2))
    * // tensor dtype=torch.float32, size=[2, 2]
    * // [[ 1.0779,  0.0383],
    * //  [-0.8785, -1.1089]]
    *
    * torch.where(x > 0, x, 0.0)
    * // tensor dtype=torch.float32, size=[2, 2]
    * // [[1.0779, 0.0383],
    * //  [0.0, 0.0]]
    * ```
    *
    * @group indexing_slicing_joining_mutating_ops
    *
    * @param condition
    *   When True (nonzero), yield input, otherwise yield other
    */
  def where[D <: DType](condition: Tensor[Bool], input: Tensor[D], other: Tensor[D]): Tensor[D] =
    fromNative(torchNative.where(condition.native, input.native, other.native))
  def where[D <: DType](condition: Tensor[Bool], input: Tensor[D], other: ScalaType): Tensor[D] =
    fromNative(torchNative.where(condition.native, input.native, other.toScalar))
  def where[D <: DType](condition: Tensor[Bool], input: ScalaType, other: Tensor[D]): Tensor[D] =
    fromNative(torchNative.where(condition.native, input.toScalar, other.native))
  def where[D <: DType](condition: Tensor[Bool], input: ScalaType, other: ScalaType): Tensor[D] =
    fromNative(torchNative.where(condition.native, input.toScalar, other.toScalar))
}
