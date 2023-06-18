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

/** Comparison Ops
  *
  * https://pytorch.org/docs/stable/torch.html#comparison-ops
  */

def allclose(
    input: Tensor[?],
    other: Tensor[?],
    rtol: Double = 1e-05,
    atol: Double = 1e-08,
    equalNan: Boolean = false
) =
  torchNative.allclose(input.native, other.native, rtol, atol, equalNan)
