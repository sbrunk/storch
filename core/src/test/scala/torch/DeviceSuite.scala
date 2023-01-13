package torch

import munit.ScalaCheckSuite
import torch.DeviceType.CUDA
import org.scalacheck.Prop.*
import org.bytedeco.pytorch.global.torch as torch_native
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck._
import Gen._
import Arbitrary.arbitrary
import DeviceType.CPU
import Generators.{*, given}

class DeviceSuite extends ScalaCheckSuite {
  test("device native roundtrip") {
    val d = Device("cpu")
    assertEquals(d, Device(d.toNative))
  }

  property("device native roundtrip for all") {
    forAll { (d: Device) =>
      assertEquals(d, Device(d.toNative))
    }
  }
}
