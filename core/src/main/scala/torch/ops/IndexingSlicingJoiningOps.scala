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
import org.bytedeco.pytorch.{TensorArrayRef, TensorVector}

/** Indexing, Slicing, Joining, Mutating Ops
  *
  * https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
  */
private[torch] trait IndexingSlicingJoiningOps {

  // TODO adjoint
  def adjoint[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.adjoint(input.native))

  // TODO argwhere
  def argwhere[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.argwhere(input.native))

  def cat[D <: DType](tensors: Seq[Tensor[D]], dim: Int = 0): Tensor[D] = Tensor(
    torchNative.cat(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)), dim.toLong)
  )

  // TODO conj
  def conjugate[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.conj(input.native))

  // TODO chunk
  def chunk[D <: DType](input: Tensor[D], chunks: Int, dim: Int = 0): Seq[Tensor[D]] = {
    val tensors = torchNative.chunk(input.native, chunks, dim.toLong)
    (0L until tensors.size()).map(i => Tensor(tensors.get(i)))
  }

  // TODO dsplit
  def dsplit[D <: DType](input: Tensor[D], indicesOrSections: Int): Seq[Tensor[D]] = {
    val tensors = torchNative.dsplit(input.native, indicesOrSections)
    (0L until tensors.size()).map(i => Tensor(tensors.get(i)))
  }
  // TODO column_stack
  def columnStack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] = Tensor(
    torchNative.column_stack(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)))
  )

  // TODO dstack
  def dstack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] = Tensor(
    torchNative.dstack(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)))
  )

  // TODO gather
  def gather[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      sparseGrad: Boolean = false
  ): Tensor[D] =
    Tensor(torchNative.gather(input.native, dim.toLong, index.native, sparseGrad))

  // TODO hsplit
  def hsplit[D <: DType](input: Tensor[D], indicesOrSections: Int): Seq[Tensor[D]] = {
    val tensors = torchNative.hsplit(input.native, indicesOrSections)
    (0L until tensors.size()).map(i => Tensor(tensors.get(i)))
  }

  // TODO hstack
  def hstack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] = Tensor(
    torchNative.hstack(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)))
  )

  // TODO index_add
  def indexAdd[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      source: Tensor[D]
  ): Tensor[D] =
    Tensor(torchNative.index_add(input.native, dim.toLong, index.native, source.native))

  // TODO index_copy
  def indexCopy[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      source: Tensor[D]
  ): Tensor[D] =
    Tensor(torchNative.index_copy(input.native, dim.toLong, index.native, source.native))

  // TODO index_reduce
  // TODO Enum for reduce: String
  // def indexReduce[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64], reduce: String): Tensor[D] =
  //   Tensor(torchNative.index_reduce(input.native, dim.toLong, index.native))

  // TODO index_select
  def indexSelect[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64]): Tensor[D] =
    Tensor(torchNative.index_select(input.native, dim.toLong, index.native))

  // TODO masked_select
  def maskedSelect[D <: DType](input: Tensor[D], mask: Tensor[Bool]): Tensor[D] =
    Tensor(torchNative.masked_select(input.native, mask.native))

  // TODO movedim
  def movedim[D <: DType](input: Tensor[D], source: Int, destination: Int): Tensor[D] =
    Tensor(torchNative.movedim(input.native, source.toLong, destination.toLong))

  // TODO moveaxis
  // TODO source, destination can be tuples
  def moveaxis[D <: DType](input: Tensor[D], source: Int, destination: Int): Tensor[D] =
    Tensor(torchNative.moveaxis(input.native, source.toLong, destination.toLong))

  // TODO narrow
  def narrow[D <: DType](input: Tensor[D], dim: Int, start: Int, length: Int): Tensor[D] =
    Tensor(torchNative.narrow(input.native, dim.toLong, start.toLong, length.toLong))

  // TODO narrow_copy
  def narrowCopy[D <: DType](input: Tensor[D], dim: Int, start: Int, length: Int): Tensor[D] =
    Tensor(torchNative.narrow_copy(input.native, dim.toLong, start.toLong, length.toLong))

  // TODO nonzero
  def nonzero[D <: DType](input: Tensor[D]): Tensor[Int64] = Tensor(
    torchNative.nonzero(input.native)
  )

  // TODO permute
  // TODO Int* vs Tuple
  def permute[D <: DType](input: Tensor[D], dims: Int*): Tensor[D] =
    Tensor(torchNative.permute(input.native, dims.map(_.toLong)*))

  // TODO reshape
  // TODO Int* vs Tuple
  def reshape[D <: DType](input: Tensor[D], shape: Int*): Tensor[D] =
    Tensor(torchNative.reshape(input.native, shape.map(_.toLong)*))

  // TODO select
  def select[D <: DType](input: Tensor[D], dim: Int, index: Int): Tensor[D] =
    Tensor(torchNative.select(input.native, dim.toLong, index.toLong))

  // TODO scatter
  // TODO reduction arg
  def scatter[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      source: Tensor[D]
  ): Tensor[D] =
    Tensor(torchNative.scatter(input.native, dim.toLong, index.native, source.native))

  // TODO diagonal_scatter
  def diagonalScatter[D <: DType](
      input: Tensor[D],
      src: Tensor[D],
      offset: Int = 0,
      dim1: Int = 0,
      dim2: Int = 1
  ): Tensor[D] =
    Tensor(
      torchNative.diagonal_scatter(
        input.native,
        src.native,
        offset.toLong,
        dim1.toLong,
        dim2.toLong
      )
    )

  // TODO select_scatter
  def selectScatter[D <: DType](input: Tensor[D], src: Tensor[D], dim: Int, index: Int): Tensor[D] =
    Tensor(torchNative.select_scatter(input.native, src.native, dim.toLong, index.toLong))

  // TODO slice_scatterd
  // TODO Review default start and end
  def sliceScatter[D <: DType](
      input: Tensor[D],
      src: Tensor[D],
      dim: Int,
      start: Int | Option[Int] = None,
      end: Int | Option[Int] = None,
      step: Int = 1
  ): Tensor[D] =
    Tensor(
      torchNative.slice_scatter(
        input.native,
        src.native,
        dim.toLong,
        start.toOptional,
        end.toOptional,
        step.toLong
      )
    )

  // TODO scatter_add
  def scatterAdd[D <: DType](
      input: Tensor[D],
      dim: Int,
      index: Tensor[Int64],
      src: Tensor[D]
  ): Tensor[D] =
    Tensor(torchNative.scatter_add(input.native, dim.toLong, index.native, src.native))

  // TODO scatter_reduce
  // TODO enum for reduce options?
  // def scatterReduce[D <: DType](input: Tensor[D], dim: Int, index: Tensor[Int64], src: Tensor[D], reduce: String, includeSelf: Boolean): Tensor[D] =
  //   Tensor(torchNative.scatter_reduce(input.native, dim.toLong, index.native, src.native, reduce))

  // TODO split
  def split[D <: DType](
      input: Tensor[D],
      splitSizeOrSections: Int,
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result = torchNative.split(input.native, splitSizeOrSections.toLong, dim.toLong)
    (0L until result.size()).map(i => Tensor(result.get(i)))
  }

  // TODO squeeze
  // TODO Review nativeDim default
  def squeeze[D <: DType](input: Tensor[D], dim: Int*): Tensor[D] = {
    Tensor(torchNative.squeeze(input.native, dim.map(_.toLong)*))
  }

  /** Concatenates a sequence of tensors along a new dimension.
    *
    * All tensors need to be of the same size.
    */
  def stack[D <: DType](tensors: Seq[Tensor[D]], dim: Int = 0): Tensor[D] = Tensor(
    torchNative.stack(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)), dim)
  )

  // TODO swapaxes
  def swapaxes[D <: DType](input: Tensor[D], axis1: Int, axis2: Int): Tensor[D] =
    Tensor(torchNative.swapaxes(input.native, axis1.toLong, axis2.toLong))

  // TODO swapdims
  def swapdims[D <: DType](input: Tensor[D], axis1: Int, axis2: Int): Tensor[D] =
    Tensor(torchNative.swapdims(input.native, axis1.toLong, axis2.toLong))

  // TODO t
  def t[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.t(input.native))

  // TODO take
  def take[D <: DType](input: Tensor[D], index: Tensor[Int64]): Tensor[D] =
    Tensor(torchNative.take(input.native, index.native))

  // TODO take_along_dim
  def takeAlongDim[D <: DType](
      input: Tensor[D],
      index: Tensor[Int64],
      dim: Int | Option[Int] = None
  ): Tensor[D] =
    Tensor(torchNative.take_along_dim(input.native, index.native, dim.toOptional))

  // TODO tensor_split
  def tensorSplit[D <: DType](
      input: Tensor[D],
      splitSizeOrSections: Int,
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result = torchNative.tensor_split(input.native, splitSizeOrSections.toLong, dim.toLong)
    (0L until result.size()).map(i => Tensor(result.get(i)))
  }

  // TODO tile
  def tile[D <: DType](input: Tensor[D], reps: Int*): Tensor[D] =
    Tensor(torchNative.tile(input.native, reps.map(_.toLong)*))

  // TODO transpose
  def transpose[D <: DType](input: Tensor[D], dim1: Int, dim2: Int): Tensor[D] =
    Tensor(torchNative.transpose(input.native, dim1.toLong, dim2.toLong))

  // TODO unbind
  def unbind[D <: DType](input: Tensor[D], dim: Int = 0): Seq[Tensor[D]] = {
    val result = torchNative.unbind(input.native, dim.toLong)
    (0L until result.size()).map(i => Tensor(result.get(i)))
  }

  // TODO unsqueeze
  def unsqueeze[D <: DType](input: Tensor[D], dim: Int): Tensor[D] =
    Tensor(torchNative.unsqueeze(input.native, dim.toLong))

  // TODO vsplit
  def vsplit[D <: DType](input: Tensor[D], splitSizeOrSections: Int): Seq[Tensor[D]] = {
    val result = torchNative.vsplit(input.native, splitSizeOrSections.toLong)
    (0L until result.size()).map(i => Tensor(result.get(i)))
  }

  // TODO vstack
  def vstack[D <: DType](tensors: Seq[Tensor[D]]): Tensor[D] = Tensor(
    torchNative.vstack(new TensorArrayRef(new TensorVector(tensors.map(_.native)*)))
  )

  // TODO where
  def where[D <: DType](condition: Tensor[Bool], input: Tensor[D], other: Tensor[D]): Tensor[D] =
    Tensor(torchNative.where(condition.native, input.native, other.native))
}
