package torch
package nn
package functional

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.toOptional

def maxPool2d[D <: DType](input: Tensor[D], kernelSize: Long | (Long, Long)): Tensor[D] =
  val kernelSizeNative = kernelSize match
    case (h, w): (Long, Long) => Array(h, w)
    case x: Long              => Array(x, x)
  Tensor(torchNative.max_pool2d(input.native, kernelSizeNative*))
