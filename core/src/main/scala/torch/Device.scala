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
import scala.collection.immutable.ArraySeq

enum DeviceType:
  case CPU, CUDA, MKLDNN, OPENGL, OPENCL, IDEEP, HIP, FPGA, ORT, XLA, Vulkan, Metal, XPU, MPS, Meta,
    HPU, VE, Lazy, IPU, MTIA, PrivateUse1, COMPILE_TIME_MAX_DEVICE_TYPES

object DeviceType:
  val deviceTypesLowerCase: Seq[String] =
    ArraySeq.unsafeWrapArray(DeviceType.values).map(_.toString.toLowerCase)
  def apply(v: String): DeviceType =
    val index = deviceTypesLowerCase.indexOf(v)
    if index == -1 then DeviceType.valueOf(v)
    else DeviceType.fromOrdinal(index)

case class Device(device: DeviceType, index: Byte = -1):
  private[torch] def toNative: pytorch.Device = pytorch.Device(device.ordinal.toByte, index)
object Device:
  def apply(device: String, index: Byte): Device = Device(DeviceType(device), index)
  def apply(device: String): Device = Device(device, -1: Byte)
  private[torch] def apply(native: pytorch.Device): Device =
    Device(DeviceType.fromOrdinal(native.`type`().value), native.index())
  val CPU = Device(DeviceType.CPU)
  val CUDA = Device(DeviceType.CUDA)
