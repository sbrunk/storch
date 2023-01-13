package torch
package nn
package functional
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.toOptional

def dropout[D <: DType](input: Tensor[D], p: Double = 0.5, training: Boolean = true): Tensor[D] = Tensor(
  torchNative.dropout(input.native, p, training)
)

