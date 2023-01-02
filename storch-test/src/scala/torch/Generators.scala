package torch

import org.scalacheck.Gen
import org.scalacheck.Arbitrary

object Generators:
  val genDeviceType: Gen[DeviceType] = Gen.oneOf(DeviceType.values)
  val genIndex: Gen[Byte]            = Gen.chooseNum(-1, Byte.MaxValue)
  val genCpuIndex: Gen[Byte]         = Gen.chooseNum[Byte](-1, 0)
  val genDevice: Gen[Device] = for
    deviceType <- genDeviceType // Arbitrary(genDeviceType).arbitrary
    i          <- if deviceType == DeviceType.CPU then genCpuIndex else genIndex
  yield Device(deviceType, i)
  val genDimSize          = Gen.choose(0L, 30L)
  val genTensorSize       = Gen.choose(0, 5).flatMap(listSize => Gen.listOfN(listSize, genDimSize))
  given Arbitrary[Device] = Arbitrary(genDevice)

  val genDType = Gen.oneOf(
    int8,
    uint8,
    int16,
    int32,
    int64,
    float32,
    float64,
    complex32,
    complex64,
    complex128,
    bool,
    // qint8,
    // quint8,
    // qint32,
    bfloat16,
    // quint4x2,
    float16,
    // undefined,
    // numoptions
  )
  given Arbitrary[DType] = Arbitrary(genDType)
