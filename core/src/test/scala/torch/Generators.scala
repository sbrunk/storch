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

import org.scalacheck.{Arbitrary, Gen}

import scala.collection.immutable.ArraySeq

object Generators:
  val genDeviceType: Gen[DeviceType] = Gen.oneOf(ArraySeq.unsafeWrapArray(DeviceType.values))
  val genIndex: Gen[Byte] = Gen.chooseNum(-1, Byte.MaxValue)
  val genCpuIndex: Gen[Byte] = Gen.chooseNum[Byte](-1, 0)
  val genDevice: Gen[Device] = for
    deviceType <- genDeviceType // Arbitrary(genDeviceType).arbitrary
    i <- if deviceType == DeviceType.CPU then genCpuIndex else genIndex
  yield Device(deviceType, i)
  val genDimSize = Gen.choose(0, 30)
  val genTensorSize = Gen.choose(0, 5).flatMap(listSize => Gen.listOfN(listSize, genDimSize))
  given Arbitrary[Device] = Arbitrary(genDevice)

  val allDTypes: List[DType] = List(
    int8,
    uint8,
    int16,
    int32,
    int64,
    float32,
    float64,
    // complex32, // NOTE: A lot of CPU operations do not support this dtype yet
    complex64,
    complex128,
    bool,
    // qint8,
    // quint8,
    // qint32,
    bfloat16
    // quint4x2,
    // float16, // NOTE: A lot of CPU operations do not support this dtype yet
    // undefined,
    // numoptions
  )

  /* This method generates tensors of multiple DTypes, and it casts them to the given concrete subtype of DType,
   * so we can use them in operations that require a specific dtype at compile time but may fail with a runtime error.
   * It is being used for property testing, and complement-property testing of tensor operations.
   */
  inline def genTensor[D <: DType](
      filterDTypes: Boolean = false,
      tensorDimensions: Int = 2
  ): Gen[Tensor[D]] =
    Gen.oneOf(allDTypes.filter(_.isInstanceOf[D] || !filterDTypes)).map { dtype =>
      ones(Seq.fill(tensorDimensions)(4), dtype = dtype.asInstanceOf[D])
    }

  val genDType = Gen.oneOf(allDTypes)
  given Arbitrary[DType] = Arbitrary(genDType)
