package torch
package nn
package functional


import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.toOptional
import org.bytedeco.pytorch.BCEWithLogitsLossOptions

// Loss functions

/** Function that measures Binary Cross Entropy between target and input logits.
  *
  * TODO support weight, reduction, pos_weight
  */
 def binaryCrossEntropyWithLogits[I <: BFloat16 | Float32 | Float64, O <: BFloat16 | Float16 | Float32 | Float64](
     input: Tensor[I],
     target: Tensor[O]
 ): Tensor[O] =
   Tensor(
     torchNative.binary_cross_entropy_with_logits(input.native, target.native, BCEWithLogitsLossOptions())
   )
