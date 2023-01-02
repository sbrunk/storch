package torch
package nn
package functional

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.toOptional

// Linear functions

def linear[D <: DType](input: Tensor[D], weight: Tensor[D], bias: Tensor[D] | Option[Tensor[D]] = None): Tensor[D] =
  Tensor(
    torchNative.linear(input.native, weight.native, toOptional(bias))
  )

def bilinear[D <: DType](
    input1: Tensor[D],
    input2: Tensor[D],
    weight: Tensor[D],
    bias: Tensor[D] | Option[Tensor[D]] = None
): Tensor[D] = Tensor(
  torchNative.bilinear(input1.native, input2.native, weight.native, toOptional(bias))
)
