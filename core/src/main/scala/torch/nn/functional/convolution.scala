package torch
package nn
package functional

import org.bytedeco.pytorch
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

/** Applies a 1D convolution over an input signal composed of several input planes. */
def conv1d[D <: FloatNN | ComplexNN](
    input: Tensor[D],
    weight: Tensor[D],
    bias: Tensor[D] | Option[Tensor[D]] = None,
    stride: Long = 1,
    padding: Long = 0,
    dilation: Long = 1,
    groups: Long = 1
): Tensor[D] =
  Tensor(
    torchNative.conv1d(
      input.native,
      weight.native,
      toOptional(bias),
      Array(stride),
      Array(padding),
      Array(dilation),
      groups
    )
  )

/** Applies a 2D convolution over an input signal composed of several input planes. */
def conv2d[D <: FloatNN | ComplexNN](
    input: Tensor[D],
    weight: Tensor[D],
    bias: Tensor[D] | Option[Tensor[D]] = None,
    stride: Long | (Long, Long) = 1,
    padding: Long | (Long, Long) = 0,
    dilation: Long | (Long, Long) = 1,
    groups: Long = 1
): Tensor[D] =
  Tensor(
    torchNative.conv2d(
      input.native,
      weight.native,
      toOptional(bias),
      toArray(stride),
      toArray(padding),
      toArray(dilation),
      groups
    )
  )
