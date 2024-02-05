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

class TensorSuite extends TensorCheckSuite {

  test("tensor properties") {
    val t = ones(Seq(2, 3), dtype = float32)
    assertEquals(t.size, Seq[Int](2, 3))
    assertEquals(t.device, Device(DeviceType.CPU, -1: Byte))
    assertEquals(t.numel, 2L * 3)
  }

  // property("tensor requiresGrad") {
  //   forAll { (dtype: FloatNN | ComplexNN, requiresGrad: Boolean) =>
  //     val t = ones(Seq(2, 3), dtype, requiresGrad=requiresGrad)
  //     assertEquals(t.dtype, dtype)
  //   }
  // }

  test("exp and log") {
    val t = Tensor(Seq(1.0, 2.0, 3.0))
    assertEquals(t.log(0), Tensor(0.0))
    assert(torch.allclose(t.log.exp, t))
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
    assert(t.grad.isEmpty)
    sum.backward()
    assert(t.grad.isDefined)
    t.grad.map { grad =>
      assertEquals(grad.dtype, float32)
      assert(grad.equal(torch.ones(Seq(3))))
    }
  }

  test("indexing") {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    // first row
    assertEquals(tensor(0), Tensor(Seq(0, 1, 2, 3)))
    // first column
    assertEquals(tensor(torch.Slice(), 0), Tensor(Seq(0, 4, 8, 12)))
    // last column
    assertEquals(tensor(---, -1), Tensor(Seq(3, 7, 11, 15)))
  }

  test("update/setter") {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    tensor(Seq(0)) = 20
    assertEquals(tensor(0), torch.full(Seq(4), 20))

    val updated = Tensor(30)
    tensor(Seq(1, 0)) = Tensor(30)
    assertEquals(tensor(1, 0), updated)

    // copy column 1 to column 0
    tensor(Seq(torch.Slice(), 1)) = tensor(torch.Slice(), 0)
    assertEquals(tensor(torch.Slice(), 1), tensor(torch.Slice(), 0))
  }

  test("Tensor creation properly handling buffers") {
    val value = 100L
    val data = Seq.fill(10000)(value)
    val tensors = 1.to(1000).map { _ =>
      Tensor(data)
    }
    assert(
      tensors.forall { t =>
        t.min().item == value &&
        t.max().item == value
      }
    )
  }

  test("repeat") {
    val t = torch.Tensor(Seq(1, 2, 3))
    val repeated = t.repeat(4, 2)

    val repeatCols = torch.cat(Seq(t, t))
    val repeatRows = torch.stack(Seq.fill(4)(repeatCols))

    assert(repeated equal repeatRows)

    assertEquals(t.repeat(4, 2, 1).size, Seq(4, 2, 3))
  }

  test("trace") {
    val t = torch.eye(3)
    assertEquals(t.trace, Tensor(3f))
  }
}
