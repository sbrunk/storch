package torch
package nn
package loss

import org.bytedeco.pytorch.CrossEntropyLossImpl
import torch.nn.modules.Module
import torch.{DType, Tensor}

/** This criterion computes the cross entropy loss between input and target. */
// TODO optional args
final class CrossEntropyLoss extends Module {
  override private[torch] val nativeModule: CrossEntropyLossImpl = CrossEntropyLossImpl()

  def apply[D <: DType](input: Tensor[D], target: Tensor[_]): Tensor[D] = Tensor(nativeModule.forward(input.native, target.native))
}
