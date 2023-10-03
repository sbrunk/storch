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
package functional

import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

private[torch] trait Sparse {

  /** Takes LongTensor with index values of shape `(*)` and returns a tensor of shape `(*,
    * numClasses)` that have zeros everywhere except where the index of last dimension matches the
    * corresponding value of the input tensor, in which case it will be 1.
    *
    * @group nn_sparse
    */
  def oneHot(input: Tensor[Int64], numClasses: Long = -1): Tensor[Int64] =
    fromNative(torchNative.one_hot(input.native, numClasses))
}
