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
package ops

class OtherOpsSuite extends TensorCheckSuite {
  test("einsum") {
    // trace
    torch.einsum("ii", torch.eye(5)).item == 5f
    val a = torch.arange(end = 25).reshape(5, 5)
    val b = torch.arange(end = 5)
    assert(torch.einsum("ii", a) equal torch.trace(a))
    // diagonal
    assert(torch.einsum("ii->i", a) equal Tensor(Seq(0, 6, 12, 18, 24)))
    // inner product
    assert(torch.einsum("i,i", b, b) equal Tensor(30))
    // matrix vector multiplication
    assert(torch.einsum("ij,j", a, b) equal Tensor(Seq(30, 80, 130, 180, 230)))
  }

  test("trace") {
    val t = torch.arange(1f, 10f).view(3, 3)
    assert(torch.trace(t) equal Tensor(15f))
  }
}
