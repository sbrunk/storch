package torch
package optim
package lr_scheduler

import org.bytedeco.pytorch

trait LRScheduler:
  def step(): Unit