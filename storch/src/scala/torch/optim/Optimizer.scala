package torch
package optim

import org.bytedeco.pytorch

/** Base class for all optimizers. */
abstract class Optimizer {
  private[torch] def native: pytorch.Optimizer

  /** Performs a single optimization step (parameter update).
    *
    * @note
    *   Unless otherwise specified, this function should not modify the ``.grad`` field of the parameters.
    */
  def step(): Unit = native.step()
  /** Sets the gradients of all optimized `Tensor`s to zero. */
  def zeroGrad(): Unit = native.zero_grad()
}
