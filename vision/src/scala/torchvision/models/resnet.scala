// derived from https://github.com/pytorch/vision/blob/v0.14.1/torchvision/models/resnet.py
// does not support downloading of pre-trained weights yet
package torchvision
package models

import torch.nn as nn
import torch.nn.HasWeight
import torch.Tensor
import torch.{Float32, Float64, BFloat16}
import torch.Float32Tensor

import torch.nn.init.{constant_, kaimingNormal_, Mode, NonLinearity}
import torch.nn.TensorModule
import torch.DType
import torch.FloatNN
import scala.collection.mutable
import torch.nn.BatchNorm2d
import sourcecode.Name
import scala.util.Using

object resnet:
  /** 3x3 convolution with padding */
  def conv3x3[D <: BFloat16 | Float32 | Float64](
      inPlanes: Int,
      outPlanes: Int,
      stride: Int = 1,
      groups: Int = 1,
      dilation: Int = 1
  ): nn.Conv2d[D] =
    nn.Conv2d[D](
      inPlanes,
      outPlanes,
      kernelSize = 3,
      stride = stride,
      padding = dilation,
      groups = groups,
      bias = false,
      dilation = dilation
    )

  /** 1x1 convolution */
  def conv1x1[D <: FloatNN](inPlanes: Int, outPlanes: Int, stride: Int = 1): nn.Conv2d[D] =
    nn.Conv2d[D](inPlanes, outPlanes, kernelSize = 1, stride = stride, bias = false)

  sealed abstract class BlockBuilder:
    val expansion: Int
    def apply[D <: BFloat16 | Float32 | Float64](
        inplanes: Int,
        planes: Int,
        stride: Int = 1,
        downsample: Option[TensorModule[D]] = None,
        groups: Int = 1,
        baseWidth: Int = 64,
        dilation: Int = 1,
        normLayer: (Int => HasWeight[D] & TensorModule[D]) = (numFeatures => nn.BatchNorm2d[D](numFeatures))
    ): TensorModule[D] = this match
      case BasicBlock => new BasicBlock(inplanes, planes, stride, downsample, groups, baseWidth, dilation, normLayer)
      case Bottleneck => new Bottleneck(inplanes, planes, stride, downsample, groups, baseWidth, dilation, normLayer)

  object BasicBlock extends BlockBuilder:
    override val expansion: Int = 1

  object Bottleneck extends BlockBuilder:
    override val expansion: Int = 4

  class BasicBlock[D <: BFloat16 | Float32 | Float64](
      inplanes: Int,
      planes: Int,
      stride: Int = 1,
      downsample: Option[TensorModule[D]] = None,
      groups: Int = 1,
      baseWidth: Int = 64,
      dilation: Int = 1,
      normLayer: => (Int => TensorModule[D]) = (numFeatures => nn.BatchNorm2d[D](numFeatures))
  ) extends TensorModule[D] {
    import BasicBlock.expansion

    if groups != 1 || baseWidth != 64 then
      throw new IllegalArgumentException("BasicBlock only supports groups=1 and baseWidth=64")
    if dilation > 1 then throw new NotImplementedError("Dilation > 1 not supported in BasicBlock")
    // Both conv1 and downsample layers downsample the input when stride != 1
    val conv1 = register(conv3x3[D](inplanes, planes, stride))
    val bn1   = register(normLayer(planes))
    val relu  = register(nn.ReLU(inplace = true))
    val conv2 = register(conv3x3[D](planes, planes))
    val bn2   = register(normLayer(planes))
    downsample.foreach(downsample => register(downsample)(using Name("downsample")))

    def apply(x: Tensor[D]): Tensor[D] =
      var identity = x

      var out = conv1(x)
      out = bn1(out)
      out = relu(out)

      out = conv2(out)
      out = bn2(out)

      downsample.foreach { downsample =>
        identity = downsample(x)
      }

      out += identity
      out = relu(out)

      out
    override def toString(): String = getClass().getSimpleName()
  }

  /** Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2) while original
    * implementation places the stride at the first 1x1 convolution(self.conv1) according to "Deep residual learning for
    * image recognition"https://arxiv.org/abs/1512.03385. This variant is also known as ResNet V1.5 and improves accuracy
    * according to https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    */
  class Bottleneck[D <: BFloat16 | Float32 | Float64](
      inplanes: Int,
      planes: Int,
      stride: Int = 1,
      downsample: Option[nn.TensorModule[D]] = None,
      groups: Int = 1,
      baseWidth: Int = 64,
      dilation: Int = 1,
      normLayer: (Int => HasWeight[D] & TensorModule[D]) = (numFeatures => nn.BatchNorm2d[D](numFeatures))
  ) extends TensorModule[D]:
    import Bottleneck.expansion

    val width = (planes * (baseWidth / 64.0)).toInt * groups
    // Both self.conv2 and self.downsample layers downsample the input when stride != 1
    val conv1 = register(conv1x1[D](inplanes, width))
    val bn1   = register(normLayer(width))
    val conv2 = register(conv3x3[D](width, width, stride, groups, dilation))
    val bn2   = register(normLayer(width))
    val conv3 = register(conv1x1[D](width, planes * expansion))
    val bn3   = register(normLayer(planes * expansion))
    val relu  = register(nn.ReLU(inplace = true))
    downsample.foreach(downsample => register(downsample)(using Name("downsample")))

    def apply(x: Tensor[D]): Tensor[D] =
      var identity = x

      var out = conv1(x)
      out = bn1(out)
      out = relu(out)

      out = conv2(out)
      out = bn2(out)
      out = relu(out)

      out = conv3(out)
      out = bn3(out)

      downsample.foreach { downsample =>
        identity = downsample(x)
      }

      out += identity
      out = relu(out)

      out
    override def toString(): String = getClass().getSimpleName()

  class ResNet[D <: BFloat16 | Float32 | Float64](
      block: BlockBuilder,
      layers: Seq[Int],
      numClasses: Int = 1000,
      zeroInitResidual: Boolean = false,
      groups: Int = 1,
      widthPerGroup: Int = 64,
      // each element in the tuple indicates if we should replace
      // the 2x2 stride with a dilated convolution instead
      replaceStrideWithDilation: (Boolean, Boolean, Boolean) = (false, false, false),
      normLayer: (Int => HasWeight[D] & TensorModule[D]) = (numFeatures => nn.BatchNorm2d[D](numFeatures))
  ) extends nn.Module {
    var inplanes  = 64
    var dilation  = 1
    val baseWidth = widthPerGroup
    val conv1     = register(nn.Conv2d[D](3, inplanes, kernelSize = 7, stride = 2, padding = 3, bias = false))
    val bn1       = register(normLayer(inplanes))
    val relu      = register(nn.ReLU(inplace = true))
    val maxpool   = register(nn.MaxPool2d[D](kernelSize = 3, stride = Some(2), padding = 1))
    val layer1    = register(makeLayer(block, 64, layers(0)))
    val layer2    = register(makeLayer(block, 128, layers(1), stride = 2, dilate = replaceStrideWithDilation(0)))
    val layer3    = register(makeLayer(block, 256, layers(2), stride = 2, dilate = replaceStrideWithDilation(1)))
    val layer4    = register(makeLayer(block, 512, layers(3), stride = 2, dilate = replaceStrideWithDilation(2)))
    val avgpool   = register(nn.AdaptiveAvgPool2d((1, 1)))
    val fc        = register(nn.Linear[D](512 * block.expansion, numClasses))

    for (m <- modules)
      m match
        case m: nn.Conv2d[_] =>
          nn.init.kaimingNormal_(m.weight, mode = Mode.FanOut, nonlinearity = NonLinearity.ReLU)
        case m: nn.BatchNorm2d[_] =>
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
        case m: nn.GroupNorm[_] =>
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
        case _ =>

    // Zero-initialize the last BN in each residual branch,
    // so that the residual branch starts with zeros, and each residual block behaves like an identity.
    // This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zeroInitResidual then
      for (m <- modules)
        m match
          case m: Bottleneck[_] if m.bn3.weight.dtype != DType.undefined =>
            nn.init.constant_(m.bn3.weight, 0)
          case m: Bottleneck[_] if m.bn2.weight.dtype != DType.undefined =>
            nn.init.constant_(m.bn2.weight, 0)

    private def makeLayer(
        block: BlockBuilder,
        planes: Int,
        blocks: Int,
        stride: Int = 1,
        dilate: Boolean = false
    ): nn.Sequential[D] = {
      var downsample: Option[TensorModule[D]] = None
      val previous_dilation                   = this.dilation
      var _stride: Int                        = stride
      if dilate then
        this.dilation *= stride
        _stride = 1
      if _stride != 1 || inplanes != planes * block.expansion then
        downsample = Some(
          nn.Sequential[D](
            conv1x1(inplanes, planes * block.expansion, _stride),
            normLayer(planes * block.expansion)
          )
        )

      var layers = Vector[TensorModule[D]]()
      layers = layers :+ block(
        this.inplanes,
        planes,
        _stride,
        downsample,
        this.groups,
        this.baseWidth,
        previous_dilation,
        normLayer
      )

      inplanes = planes * block.expansion
      for (_ <- 1 until blocks)
        layers = layers :+
          block(
            this.inplanes,
            planes,
            groups = this.groups,
            baseWidth = this.baseWidth,
            dilation = this.dilation,
            normLayer = normLayer
          )

      nn.Sequential[D](layers*)
    }

    private def forwardImpl(_x: Tensor[D]): Tensor[D] =
      var x = conv1(_x)
      x = bn1(x)
      x = relu(x)
      x = maxpool(x)

      x = layer1(x)
      x = layer2(x)
      x = layer3(x)
      x = layer4(x)

      x = avgpool(x)
      x = x.flatten(1)
      fc(x)

    def apply(x: Tensor[D]): Tensor[D] =
      forwardImpl(x)
  }

  /** ResNet-18 from [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). */
  def resnet18[D <: BFloat16 | Float32 | Float64](numClasses: Int = 1000) =
    ResNet[D](BasicBlock, Seq(2, 2, 2, 2), numClasses)

  /** ResNet-34 from [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). */
  def resnet34[D <: BFloat16 | Float32 | Float64](numClasses: Int = 1000) =
    ResNet[D](BasicBlock, Seq(3, 4, 6, 3), numClasses = numClasses)

  /** ResNet-50 from [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) */
  def resnet50[D <: BFloat16 | Float32 | Float64](numClasses: Int = 1000) =
    ResNet[D](Bottleneck, Seq(3, 4, 6, 3), numClasses = numClasses)

  /** ResNet-101 from [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) */
  def resnet101[D <: BFloat16 | Float32 | Float64](numClasses: Int = 1000) =
    ResNet[D](Bottleneck, Seq(3, 4, 23, 3), numClasses = numClasses)

  /** ResNet-152 from [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) */
  def resnet152[D <: BFloat16 | Float32 | Float64](numClasses: Int = 1000) =
    ResNet[D](Bottleneck, Seq(3, 8, 36, 3), numClasses = numClasses)

  // TODO ResNeXt and wide ResNet variants