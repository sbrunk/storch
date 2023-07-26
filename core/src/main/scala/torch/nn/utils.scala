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
package nn

import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.TensorVector

object utils:
  def clipGradNorm_(
      parameters: Seq[Tensor[?]],
      max_norm: Double,
      norm_type: Double = 2.0,
      error_if_nonfinite: Boolean = false
  ): Double =
    torchNative.clip_grad_norm_(
      TensorVector(parameters.map(_.native).toArray*),
      max_norm,
      norm_type,
      error_if_nonfinite
    )
