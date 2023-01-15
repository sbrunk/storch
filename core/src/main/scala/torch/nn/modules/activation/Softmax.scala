package torch.nn.modules.activation

import org.bytedeco.pytorch.SoftmaxImpl
import torch.nn.modules.Module
import torch.{DType, Tensor}

/** Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the
  * n-dimensional output Tensor lie in the range [0,1] and sum to 1.
  *
  * Softmax is defined as: $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
  *
  * When the input Tensor is a sparse tensor then the unspecifed values are treated as ``-inf``.
  */
final class Softmax(dim: Int) extends Module:
  override val nativeModule: SoftmaxImpl         = SoftmaxImpl(dim)
  def apply[D <: DType](t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))
