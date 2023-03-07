package torchvision
package transforms

import com.sksamuel.scrimage.ImmutableImage
import com.sksamuel.scrimage.ScaleMethod
import torch.Int32
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
            ScaleMethod.Bilinear
          )
        else
          image.scaleTo(
            resizeSize,
            (resizeSize * (image.height / image.width.toDouble)).toInt,
            ScaleMethod.Bilinear
          )
      val croppedImage = scaledImage.resizeTo(cropSize, cropSize)
      toTensor(croppedImage)

    def batchTransforms(input: Tensor[Float32]): Tensor[Float32] =
      torchvision.transforms.functional.normalize(
        input,
        mean = Seq(0.485f, 0.456f, 0.406f),
        std = Seq(0.229f, 0.224f, 0.225f)
      )
