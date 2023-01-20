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

package torchvision.transforms

import torch.*

object functional:

  def isTensorATorchImage(x: Tensor[?]): Boolean = x.dim >= 2

  def assertImageTensor(img: Tensor[?]): Unit =
    if !isTensorATorchImage(img) then
      throw new IllegalArgumentException("Tensor is not a torch image.")

  def normalize[D <: FloatNN](tensor: Tensor[D], mean: Seq[Float], std: Seq[Float]): Tensor[D] =
    assertImageTensor(tensor)

    if tensor.dim < 3 then
      throw new IllegalArgumentException(
        s"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = ${tensor.size}"
      )

    val dtype = tensor.dtype
    var _mean = Tensor(mean, device = tensor.device).to(dtype = dtype)
    var _std = Tensor(std, device = tensor.device).to(dtype = dtype)
    if (_std == 0).any.item then
      throw new IllegalArgumentException(
        f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
      )
    if _std.dim == 1 then _mean = _mean.view(-1, 1, 1)
    if _std.dim == 1 then _std = _std.view(-1, 1, 1)
    (tensor - _mean) / _std
