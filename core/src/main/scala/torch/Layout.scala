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
import org.bytedeco.pytorch.global.torch as torchNative

/** A `torch.layout` is an object that represents the memory layout of a torch.Tensor.
  *
  * Currently, we support ``torch.strided`` (dense Tensors) and have beta support for
  * ``torch.sparse_coo`` (sparse COO Tensors).
  *
  * torch.strided represents dense Tensors and is the memory layout that is most commonly used. Each
  * strided tensor has an associated torch.Storage, which holds its data. These tensors provide
  * multi-dimensional, strided view of a storage. Strides are a list of integers: the k-th stride
  * represents the jump in the memory necessary to go from one element to the next one in the k-th
  * dimension of the Tensor. This concept makes it possible to perform many tensor operations
  * efficiently.
  */
enum Layout:
  case Strided, Sparse, SparseCsr, Mkldnn, NumOptions
  private[torch] def toNative: torchNative.Layout = torchNative.Layout.valueOf(this.toString)

object Layout:
  private[torch] def fromNative(native: torchNative.Layout) = Layout.valueOf(native.toString)
