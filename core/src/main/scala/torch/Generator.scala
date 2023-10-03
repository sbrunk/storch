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

import org.bytedeco.pytorch.global.torch.make_generator_cpu
import org.bytedeco.pytorch.global.torch_cuda.make_generator_cuda
import torch.internal.NativeConverters.fromNative

/** Creates and returns a generator object that manages the state of the algorithm which produces
  * pseudo random numbers.
  */
class Generator(val device: Device = Device.CPU) {
  private[torch] val native = device.device match
    case DeviceType.CPU  => make_generator_cpu
    case DeviceType.CUDA => make_generator_cuda
    case _               => throw new IllegalArgumentException("Unsupported generator device")

  /** Returns the Generator state as a [[torch.Tensor[UInt8]]. */
  def getState: Tensor[UInt8] = fromNative(native.get_state())

  /** Sets the Generator state.
    *
    * @param newState
    *   The desired state.
    */
  def setState(newState: Tensor[UInt8]) = native.set_state(newState.native)

  /** Returns the initial seed for generating random numbers. */
  def initialSeed: Long = native.seed()

  /** Sets the seed for generating random numbers. Returns a torch.Generator object.
    *
    * It is recommended to set a large seed, i.e. a number that has a good balance of 0 and 1 bits.
    * Avoid having many 0 bits in the seed.
    *
    * @param seed
    *   The desired seed. Value must be within the inclusive range
    *   *[-0x8000_0000_0000_0000,0xffff_ffff_ffff_ffff]*. Otherwise, a RuntimeError is raised.
    *   Negative inputs are remapped to positive values with the *formula 0xffff_ffff_ffff_ffff +
    *   seed*.
    */
  def manualSeed(seed: Long): Unit = native.set_current_seed(seed)

  /** Gets a non-deterministic random number from std::random_device or the current time and uses it
    * to seed a Generator.
    */
  def seed: Long = native.current_seed()
}
