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
import org.bytedeco.pytorch.{AdamWOptions, TensorVector}

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
