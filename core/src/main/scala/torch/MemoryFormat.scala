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

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative

/** A memoryFormat is an object representing the memory format on which a torch.Tensor is or will be
  * allocated.
  */
enum MemoryFormat:
  /** Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values
    * in decreasing order.
    */
  case Contiguous

  /** Used in functions like clone to preserve the memory format of the input tensor. If input
    * tensor is allocated in dense non-overlapping memory, the output tensor strides will be copied
    * from the input. Otherwise output strides will follow torch.contiguous_format
    */
  case Preserve

  /** Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values
    * in `strides[0] > strides[2] > strides[3] > strides[1] == 1` aka NHWC order.
    */
  case ChannelsLast
  case ChannelsLast3d

  private[torch] def toNative: torchNative.MemoryFormat =
    torchNative.MemoryFormat.valueOf(this.toString)
  private[torch] def toNativeOptional: pytorch.MemoryFormatOptional =
    pytorch.MemoryFormatOptional(torchNative.MemoryFormat.valueOf(this.toString))
