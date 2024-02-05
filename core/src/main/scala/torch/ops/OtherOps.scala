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
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.pytorch.global.torch as torchNative

/** Other Ops
  *
  * https://pytorch.org/docs/stable/torch.html#other-operations
  */
private[torch] trait OtherOps {

  /** Sums the product of the elements of the input `operands` along dimensions specified using a
    * notation based on the Einstein summation convention.
    *
    * Einsum allows computing many common multi-dimensional linear algebraic array operations by
    * representing them in a short-hand format based on the Einstein summation convention, given by
    * `equation`. The details of this format are described below, but the general idea is to label
    * every dimension of the input `operands` with some subscript and define which subscripts are
    * part of the output. The output is then computed by summing the product of the elements of the
    * `operands` along the dimensions whose subscripts are not part of the output. For example,
    * matrix multiplication can be computed using einsum as [torch.einsum(\"ij,jk-\>ik\", A, B)].
    * Here, j is the summation subscript and i and k the output subscripts (see section below for
    * more details on why).
    *
    * Equation:
    *
    * The `equation` string specifies the subscripts (letters in [\[a-zA-Z\]]) for each dimension of
    * the input `operands` in the same order as the dimensions, separating subscripts for each
    * operand by a comma (\',\'), e.g. [\'ij,jk\'] specify subscripts for two 2D operands. The
    * dimensions labeled with the same subscript must be broadcastable, that is, their size must
    * either match or be [1]. The exception is if a subscript is repeated for the same input
    * operand, in which case the dimensions labeled with this subscript for this operand must match
    * in size and the operand will be replaced by its diagonal along these dimensions. The
    * subscripts that appear exactly once in the `equation` will be part of the output, sorted in
    * increasing alphabetical order. The output is computed by multiplying the input `operands`
    * element-wise, with their dimensions aligned based on the subscripts, and then summing out the
    * dimensions whose subscripts are not part of the output.
    *
    * Optionally, the output subscripts can be explicitly defined by adding an arrow (\'-\>\') at
    * the end of the equation followed by the subscripts for the output. For instance, the following
    * equation computes the transpose of a matrix multiplication: \'ij,jk-\>ki\'. The output
    * subscripts must appear at least once for some input operand and at most once for the output.
    *
    * Ellipsis (\'\...\') can be used in place of subscripts to broadcast the dimensions covered by
    * the ellipsis. Each input operand may contain at most one ellipsis which will cover the
    * dimensions not covered by subscripts, e.g. for an input operand with 5 dimensions, the
    * ellipsis in the equation [\'ab\...c\'] cover the third and fourth dimensions. The ellipsis
    * does not need to cover the same number of dimensions across the `operands` but the \'shape\'
    * of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the
    * output is not explicitly defined with the arrow (\'-\>\') notation, the ellipsis will come
    * first in the output (left-most dimensions), before the subscript labels that appear exactly
    * once for the input operands. e.g. the following equation implements batch matrix
    * multiplication [\'\...ij,\...jk\'].
    *
    * A few final notes: the equation may contain whitespaces between the different elements
    * (subscripts, ellipsis, arrow and comma) but something like [\'. . .\'] is not valid. An empty
    * string [\'\'] is valid for scalar operands.
    *
    * Note: Sublist format it not supported yet
    *
    * Example:
    * ```scala sc
    * import torch.*
    * // trace
    * torch.einsum("ii", torch.randn(Seq(4, 4)))
    *
    * // diagonal
    * torch.einsum("ii->i", torch.randn(Seq(4, 4)))
    *
    * // outer product
    * val x = torch.randn(Seq(5))
    * val y = torch.randn(Seq(4))
    * torch.einsum("i,j->ij", x, y)
    *
    * // batch matrix multiplication
    * val As = torch.randn(Seq(3, 2, 5))
    * val Bs = torch.randn(Seq(3, 5, 4))
    * torch.einsum("bij,bjk->bik", As, Bs)
    *
    * // with sublist format and ellipsis
    * // Not supported yet in Storch
    * // torch.einsum(As, Seq(---, 0, 1), Bs, Seq(---, 1, 2), Seq(---, 0, 2))
    *
    * // batch permute
    * val A = torch.randn(Seq(2, 3, 4, 5))
    * torch.einsum("...ij->...ji", A).shape
    * ```
    *
    * ```scala sc
    * // equivalent to torch.nn.functional.bilinear
    * val A = torch.randn(Seq(3, 5, 4))
    * val l = torch.randn(Seq(2, 5))
    * val r = torch.randn(Seq(2, 4))
    * torch.einsum("bn,anm,bm->ba", l, A, r)
    * ```
    *
    * @group other_ops
    * @param equation
    *   The subscripts for the Einstein summation.
    * @param operands
    *   The tensors to compute the Einstein summation of.
    */
  def einsum[D <: DType](equation: String, operands: Tensor[D]*): Tensor[D] =
    // TODO the equation input is not yet working, see https://github.com/bytedeco/javacpp-presets/discussions/1390
    fromNative(torchNative.einsum(BytePointer(equation), toArrayRef(operands)))

  /** Returns the sum of the elements of the diagonal of the input 2-D matrix. */
  def trace[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.trace(input.native))
}
