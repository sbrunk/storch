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

import DeviceType.CUDA

import java.nio.{IntBuffer, LongBuffer}

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

class TensorSuite extends ScalaCheckSuite {

  test("arange") {
    val t0 = arange(0, 10)
    t0.toSeq == Seq.range(0, 10)
    val t1 = arange(0, 10, 2)
    t1.toSeq == Seq.range(0, 10, 2)
  }

  test("tensor properties") {
    val t = ones(Seq(2, 3), dtype = float32)
    assertEquals(t.size, Seq[Long](2, 3))
    assertEquals(t.device, Device(DeviceType.CPU, -1: Byte))
    assertEquals(t.numel, 2L * 3)
  }

  property("tensor dtypes") {
    forAll { (dtype: DType) =>
      val t = ones(Seq(2, 3), dtype)
      assertEquals(t.dtype, dtype)
    }
  }

  // property("tensor requiresGrad") {
  //   forAll { (dtype: FloatNN | ComplexNN, requiresGrad: Boolean) =>
  //     val t = ones(Seq(2, 3), dtype, requiresGrad=requiresGrad)
  //     assertEquals(t.dtype, dtype)
  //   }
  // }

  property("tensor ones") {
    forAll(genTensorSize, genDType) { (size, dtype) =>
      val t = ones(size, dtype)
      assertEquals(t.dtype, dtype)
      assertEquals(t.size, size)
      assertEquals(t.numel, size.product)
      assertEquals(t.toSeq.length, size.product.toInt)
    }
  }

  test("ones") {
    val t = ones[Float32](Seq(2, 3))
    assertEquals(t.size, Seq[Long](2, 3))
    assertEquals(t.numel, 2L * 3)
    assertEquals(t.toSeq, Seq.fill[Float](2 * 3)(1f))
  }

  test("toBuffer") {
    val content = Seq(1, 2, 3, 4)
    val t = Tensor(content)
    val b = t.toBuffer
    val a = new Array[Int](content.length)
    b.get(a)
    assertEquals(content, a.toSeq)
  }

  test("+") {
    assertEquals((Tensor(1) + 2).item, 3)
  }

  test("grad") {
    val t = torch.ones(Seq(3)) * 2
    t.requiresGrad = true
    val sum = t.sum
    assert(t.grad.dtype == undefined)
    sum.backward()
    assert(t.grad.dtype == float32)
    assert(t.grad.equal(torch.ones(Seq(3))))
  }
}
