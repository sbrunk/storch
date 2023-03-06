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
import com.sksamuel.scrimage.ImmutableImage
import scala.collection.immutable.ArraySeq

object functional:

  private def isTensorATorchImage(x: Tensor[?]): Boolean = x.dim >= 2

  private def assertImageTensor(img: Tensor[?]): Unit =
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
    if _mean.dim == 1 then _mean = _mean.view(-1, 1, 1)
    if _std.dim == 1 then _std = _std.view(-1, 1, 1)
    (tensor - _mean) / _std

  /** Convert an [[ImmutableImage]] (H x W x C) to a [[Tensor[Float32]] of shape (C x H x W) in the
    * range `[0.0, 1.0]`.
    */
  def toTensor(pic: ImmutableImage): Tensor[Float32] =
    val bytes = pic.rgb.flatten
    // transpose NxHxWxC to NxCxHxW because pytorch expects channels first
    Tensor(ArraySeq.unsafeWrapArray(bytes))
      .reshape(pic.height, pic.width, 3)
      .permute(2, 0, 1)
      .to(dtype = float32) / 255

  def toImmutableImage[D <: FloatNN](pic: Tensor[D]): ImmutableImage =
    var _pic = pic
    if !Seq(2, 3).contains(pic.dim) then
      throw new IllegalArgumentException(
        s"pic should be 2/3 dimensional. Got ${pic.dim} dimensions."
      )
    else if pic.dim == 2 then
      // if 2D image, add channel dimension (CHW)
      _pic = pic.unsqueeze(0)
    // check number of channels
    if pic.shape(-3) > 4 then
      throw new IllegalArgumentException(
        s"pic should not have > 4 channels. Got ${pic.shape(-3)} channels."
      )
    val intImage = (_pic.permute(1, 2, 0) * 255).to(dtype = int8)
    val bytes = intImage.toArray
    ImmutableImage.loader().fromBytes(bytes)
