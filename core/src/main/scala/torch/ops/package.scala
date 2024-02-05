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

import internal.NativeConverters.{fromNative, tensorOptions}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.MemoryFormatOptional

package object ops {

  private[torch] def xLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2,
      layout: Layout | Derive,
      device: Device | Derive,
      requiresGrad: Boolean,
      memoryFormat: MemoryFormat,
      nativeFn: (
          pytorch.Tensor,
          pytorch.TensorOptions,
          pytorch.MemoryFormatOptional
      ) => pytorch.Tensor
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] = {
    val derivedDType = dtype match
      case _: Derive => input.dtype
      case d: DType  => d
    val derivedLayout = layout match
      case _: Derive => input.layout
      case l: Layout => l
    val derivedDevice = device match
      case _: Derive => input.device
      case d: Device => d
    fromNative(
      nativeFn(
        input.native,
        tensorOptions(derivedDType, derivedLayout, derivedDevice, requiresGrad),
        new MemoryFormatOptional(memoryFormat.toNative)
      )
    )
  }
}
