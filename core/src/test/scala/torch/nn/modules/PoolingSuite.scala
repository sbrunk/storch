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

class AdapativeAvgPool2dSuite extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val m1 = AdaptiveAvgPool2d((5, 7))
    val input = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(m1(input).shape, Seq(1, 64, 5, 7))
    val m2 = nn.AdaptiveAvgPool2d((1, 1))
    assertEquals(m2(input).shape, Seq(1, 64, 1, 1))
  }
}
