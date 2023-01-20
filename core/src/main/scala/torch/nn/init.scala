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
import org.bytedeco.pytorch.FanModeType
import org.bytedeco.pytorch.kFanIn
import org.bytedeco.pytorch.kFanOut
import org.bytedeco.pytorch.NonlinearityType
import org.bytedeco.pytorch.kLinear
import org.bytedeco.pytorch.kConv1D
import org.bytedeco.pytorch.kConv2D
import org.bytedeco.pytorch.kConv3D
import org.bytedeco.pytorch.kConvTranspose1D
import org.bytedeco.pytorch.kConvTranspose2D
import org.bytedeco.pytorch.kConvTranspose3D
import org.bytedeco.pytorch.kSigmoid
import org.bytedeco.pytorch.kReLU
import org.bytedeco.pytorch.kLeakyReLU
import org.bytedeco.pytorch.Scalar

// TODO implement remaining init functions
object init:
  def kaimingNormal_(
      t: Tensor[?],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): Unit =
    torchNative.kaiming_normal_(t.native, a, mode.toNative, nonlinearity.toNative)

  enum Mode:
    case FanIn, FanOut
    private[torch] def toNative: FanModeType = FanModeType(this match
      case Mode.FanIn  => kFanIn()
      case Mode.FanOut => kFanOut()
    )

  enum NonLinearity:
    case Linear, Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D, Sigmoid,
      ReLU, LeakyReLU
    private[torch] def toNative: NonlinearityType = NonlinearityType(this match
      case NonLinearity.Linear          => kLinear()
      case NonLinearity.Conv1D          => kConv1D()
      case NonLinearity.Conv2D          => kConv2D()
      case NonLinearity.Conv3D          => kConv3D()
      case NonLinearity.ConvTranspose1D => kConvTranspose1D()
      case NonLinearity.ConvTranspose2D => kConvTranspose2D()
      case NonLinearity.ConvTranspose3D => kConvTranspose3D()
      case NonLinearity.Sigmoid         => kSigmoid()
      case NonLinearity.ReLU            => kReLU()
      case NonLinearity.LeakyReLU       => kLeakyReLU()
    )

  // TODO valid for all scala types
  def constant_(t: Tensor[?], fillValue: Double) =
    torchNative.constant_(t.native, Scalar(fillValue)): Unit
