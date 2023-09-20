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

class FlattenSuite extends munit.FunSuite {
  test("Flatten") {
    val input = torch.randn(Seq(32, 1, 5, 5))
    val m1 = nn.Flatten()
    val o1 = m1(input)
    assertEquals(o1.shape, Seq(32, 25))
    assert(input.reshape(32, 25).equal(o1))
    val m2 = nn.Flatten(0, 2)
    val o2 = m2(input)
    assertEquals(o2.shape, Seq(160, 5))
    assert(input.reshape(160, 5).equal(o2))
  }
}
