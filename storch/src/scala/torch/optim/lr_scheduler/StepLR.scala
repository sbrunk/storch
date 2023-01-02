package torch
package optim
package lr_scheduler

import org.bytedeco.pytorch

/** Decays the learning rate of each parameter group by gamma every step_size epochs.
  *
  * Notice that such decay can happen simultaneously with other changes to the learning rate from outside this
  * scheduler.
  */
class StepLR(optimizer: Optimizer, step_size: Int, gamma: Float = 0.1) extends LRScheduler {
  private[torch] val native = pytorch.StepLR(optimizer.native, step_size, gamma)

  def step(): Unit = native.step()
}
