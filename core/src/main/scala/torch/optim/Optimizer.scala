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

import org.bytedeco.pytorch

/** Base class for all optimizers. */
abstract class Optimizer {
  private[torch] def native: pytorch.Optimizer

  /** Performs a single optimization step (parameter update).
    *
    * @note
    *   Unless otherwise specified, this function should not modify the ``.grad`` field of the
    *   parameters.
    */
  def step(): Unit =
    native.step()
    // TODO check what tensor is returned by step
    ()

  /** Sets the gradients of all optimized `Tensor`s to zero. */
  def zeroGrad(): Unit = native.zero_grad()
  def zeroGrad(setToNone: Boolean = true): Unit = native.zero_grad(setToNone)

}
