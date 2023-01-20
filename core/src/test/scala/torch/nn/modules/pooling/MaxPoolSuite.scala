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
package modules
package pooling

class MaxPoolSuite extends munit.FunSuite {
  test("MaxPool2d output shapes") {
    val input = torch.randn(Seq(1, 3, 244, 244))
    // pool of square window of size=3, stride=2
    val m1 = MaxPool2d[Float32](3, stride = Some(2))
    assertEquals(m1(input).shape, Seq[Long](1, 3, 121, 121))
    // pool of non-square window
    val m2 = MaxPool2d[Float32]((3, 2), stride = Some(2, 1))
    assertEquals(m2(input).shape, Seq[Long](1, 3, 121, 243))
    val m3 = MaxPool2d[Float32](3)
    assertEquals(m3(input).shape, Seq[Long](1, 3, 81, 81))
  }
}
