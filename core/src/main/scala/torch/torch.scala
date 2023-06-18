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

import org.bytedeco.pytorch.global.torch as torchNative
import scala.util.Using

def manualSeed(seed: Long) = torchNative.manual_seed(seed)

/** Disable gradient calculation for [[op]].
  *
  * Disabling gradient calculation is useful for inference, when you are sure that you will not call
  * `Tensor.backward()`. It will reduce memory consumption for computations that would otherwise
  * have `requiresGrad=true`.
  *
  * In this mode, the result of every computation will have `requiresGrad=false`, even when the
  * inputs have `requiresGrad=true`.
  *
  * This context manager is thread local; it will not affect computation in other threads.
  *
  * @param op
  */
def noGrad[A](op: => A): A = {
  import org.bytedeco.pytorch.NoGradGuard
  Using.resource(NoGradGuard()) { _ =>
    op
  }
}

def setNumThreads(threads: Int): Unit = torchNative.set_num_threads(threads)
