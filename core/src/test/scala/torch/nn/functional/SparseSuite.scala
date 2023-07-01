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
package functional

import Generators.genTensor

class SparseSuite extends TensorCheckSuite {

  // TODO Test multi-dimensional tensors
  testUnaryOp(
    op = nn.functional.oneHot(_, numClasses = 6),
    opName = "nn.functional.oneHot",
    inputTensor = Tensor(3L),
    expectedTensor = Tensor(Seq(0L, 0L, 0L, 1L, 0L, 0L)),
    // TODO Fix genTensor for cases where the tensor type is not a union, but a concrete one, such as Tensor[Int64]
    genTensor = genTensor[Int64](filterDTypes = true)
  )

}
