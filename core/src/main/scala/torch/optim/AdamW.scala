package torch.optim

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{AdamWOptions, SGDOptions, TensorVector}
import torch.{DType, Tensor}

import scala.collection.immutable.Iterable

// format: off
/** Implements the AdamW algorithm.
 *
 */
// format: on
final class AdamW(
    params: Iterable[Tensor[?]],
    lr: Double = 1e-3,
    betas: (Double, Double) = (0.9, 0.999),
    eps: Double = 1e-8,
    weightDecay: Double = 0,
    amsgrad: Boolean = false
) extends Optimizer {
  private val nativeParams: TensorVector = TensorVector(params.map(_.native).toArray*)
  private val options: AdamWOptions = AdamWOptions(lr)
  options.betas().put(Array(betas._1, betas._2)*)
  options.eps().put(eps)
  options.weight_decay().put(weightDecay)
  options.amsgrad().put(amsgrad)
  override private[torch] val native: pytorch.AdamW = pytorch.AdamW(nativeParams, options)
}
