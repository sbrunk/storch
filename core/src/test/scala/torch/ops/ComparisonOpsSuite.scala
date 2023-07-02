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

class ComparisonOpsSuite extends TensorCheckSuite {

  testUnaryOp(
    op = argsort(_),
    opName = "argsort",
    inputTensor = Tensor(Seq(1, 3, 2)),
    expectedTensor = Tensor(Seq(0L, 2L, 1L))
  )

}
