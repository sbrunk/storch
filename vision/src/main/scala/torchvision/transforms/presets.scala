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

package torchvision
package transforms

import com.sksamuel.scrimage.ImmutableImage
import com.sksamuel.scrimage.ScaleMethod
import torch.Tensor
import torch.Float32
import torchvision.transforms.functional.toTensor

object Presets:

  class ImageClassification(
      cropSize: Int,
      resizeSize: Int = 256,
      mean: Seq[Float] = Seq(0.485f, 0.456f, 0.406f),
      std: Seq[Float] = Seq(0.229f, 0.224f, 0.225f),
      interpolation: ScaleMethod = ScaleMethod.Bilinear
  ):
    def transforms(image: ImmutableImage): Tensor[Float32] =
      val scaledImage =
        if image.height < image.width then
          image.scaleTo(
            (resizeSize * (image.width / image.height.toDouble)).toInt,
            resizeSize,
            interpolation
          )
        else
          image.scaleTo(
            resizeSize,
            (resizeSize * (image.height / image.width.toDouble)).toInt,
            interpolation
          )
      val croppedImage = scaledImage.resizeTo(cropSize, cropSize)
      toTensor(croppedImage)

    def batchTransforms(input: Tensor[Float32]): Tensor[Float32] =
      torchvision.transforms.functional.normalize(
        input,
        mean = mean,
        std = std
      )
