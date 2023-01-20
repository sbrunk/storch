/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package optim
package lr_scheduler

import org.bytedeco.pytorch

/** Decays the learning rate of each parameter group by gamma every step_size epochs.
  *
  * Notice that such decay can happen simultaneously with other changes to the learning rate from
  * outside this scheduler.
  */
class StepLR(optimizer: Optimizer, step_size: Int, gamma: Float = 0.1) extends LRScheduler {
  private[torch] val native = pytorch.StepLR(optimizer.native, step_size, gamma)

  def step(): Unit = native.step()
}
