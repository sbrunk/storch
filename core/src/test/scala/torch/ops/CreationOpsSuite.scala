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

import org.scalacheck.Prop.*
import Generators.*

class CreationOpsSuite extends TensorCheckSuite {

  test("arange.unit-test") {
    val t0 = arange(0, 10)
    assertEquals(t0.toSeq, Seq.range(0, 10))
    val t1 = arange(0, 10, 2)
    assertEquals(t1.toSeq, Seq.range(0, 10, 2))
  }

  property("ones.property-test") {
    forAll(genTensorSize, genDType) { (size, dtype) =>
      val t = ones(size, dtype)
      assertEquals(t.dtype, dtype)
      assertEquals(t.size, size)
      assertEquals(t.numel, size.product.toLong)
      assertEquals(t.toSeq.length, size.product.toInt)
    }
  }

  test("ones.unit-test") {
    val t = ones[Float32](Seq(2, 3))
    assertEquals(t.size, Seq(2, 3))
    assertEquals(t.numel, 2L * 3)
    assertEquals(t.toSeq, Seq.fill[Float](2 * 3)(1f))
  }

}
