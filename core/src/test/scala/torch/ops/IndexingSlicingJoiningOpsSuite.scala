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
import spire.math.Complex

class IndexingSlicingJoiningOpsSuite extends TensorCheckSuite {

  testUnaryOp(
    op = adjoint,
    opName = "adjoint",
    inputTensor = Tensor(
      Seq(
        Seq(Complex(0.0, 0.0), Complex(1.0, 1.0)),
        Seq(Complex(2.0, 2.0), Complex(3.0, 3.0))
      )
    ),
    expectedTensor = Tensor(
      Seq(
        Seq(Complex(0.0, -0.0), Complex(2.0, -2.0)),
        Seq(Complex(1.0, -1.0), Complex(3.0, -3.0))
      )
    )
  )

  testUnaryOp(
    op = argwhere,
    opName = "argwhere",
    inputTensor = Tensor(Seq(1, 0, 1)),
    expectedTensor = Tensor(Seq(Seq(0L), Seq(2L)))
  )

  testUnaryOp(
    op = (t: Tensor[?]) => cat(Seq(t, t), 0),
    opName = "cat",
    inputTensor = Tensor(Seq(Seq(1, 2, 3))),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3), Seq(1, 2, 3)))
  )

  testUnaryOp(
    op = conj,
    opName = "conj",
    inputTensor = Tensor(Seq(Complex(-1.0, 1.0), Complex(-2.0, 2.0), Complex(3.0, -3.0))),
    expectedTensor = Tensor(Seq(Complex(-1.0, -1.0), Complex(-2.0, -2.0), Complex(3.0, 3.0)))
  )

  propertyTestUnaryOp(op = chunk(_, 6), opName = "chunk")
  test("chunk.unit-test") {
    val inputTensor = torch.arange(end = 11)
    val expectedTensors = Seq(
      Seq(0, 1),
      Seq(2, 3),
      Seq(4, 5),
      Seq(6, 7),
      Seq(8, 9),
      Seq(10)
    ).map(Tensor.apply(_))
    val outputTensors = torch.chunk(inputTensor, chunks = 6)
    assert(
      outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
        outputTensor.allclose(expectedTensor)
      }
    )
  }

  propertyTestUnaryOp(
    op = dsplit(_, 2),
    opName = "dsplit",
    genTensor = genTensor(tensorDimensions = 3)
  )
  test("dsplit.unit-test") {
    val inputTensor = torch.arange(end = 16.0, dtype = float64).reshape(2, 2, 4)
    val expectedTensors = Seq(
      Tensor(Seq(0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0)).reshape(2, 2, 2),
      Tensor(Seq(2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0)).reshape(2, 2, 2)
    )
    val outputTensors = torch.dsplit(inputTensor, indicesOrSections = 2)
    assert(
      outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
        outputTensor.allclose(expectedTensor)
      }
    )
  }

  propertyTestUnaryOp(op = (t: Tensor[?]) => columnStack(Seq(t, t)), opName = "columnStack")
  test("columnStack.unit-test") {
    val inputTensors = Seq(Tensor(Seq(1, 2, 3)), Tensor(Seq(4, 5, 6)))
    val expectedTensor = Tensor(
      Seq(
        Seq(1, 4),
        Seq(2, 5),
        Seq(3, 6)
      )
    )
    val outputTensor = torch.columnStack(inputTensors)
    assert(outputTensor.allclose(expectedTensor))
  }

  propertyTestUnaryOp(op = (t: Tensor[?]) => dstack(Seq(t, t)), opName = "dstack")
  test("dstack.unit-test") {
    val inputTensors = Seq(
      Tensor(Seq(1, 2, 3)),
      Tensor(Seq(4, 5, 6))
    )
    val expectedTensor = Tensor(
      Seq(
        Seq(
          Seq(1, 4),
          Seq(2, 5),
          Seq(3, 6)
        )
      )
    )
    val outputTensor = torch.dstack(inputTensors)
    assert(outputTensor.allclose(expectedTensor))
  }

  val gatherIndex = Tensor(Seq(Seq(0L, 0L), Seq(1L, 0L)))
  testUnaryOp(
    op = gather(_, 1, gatherIndex),
    opName = "gather",
    inputTensor = Tensor(Seq(Seq(1, 2), Seq(3, 4))),
    expectedTensor = Tensor(Seq(Seq(1, 1), Seq(4, 3)))
  )

  propertyTestUnaryOp(op = hsplit(_, 2), opName = "hsplit")
  test("hsplit.unit-test") {
    val inputTensor = torch.arange(end = 16.0, dtype = float64).reshape(4, 4)
    val expectedTensors = Seq(
      Tensor(Seq(Seq(0.0, 1.0), Seq(4.0, 5.0), Seq(8.0, 9.0), Seq(12.0, 13.0))),
      Tensor(Seq(Seq(2.0, 3.0), Seq(6.0, 7.0), Seq(10.0, 11.0), Seq(14.0, 15.0)))
    )
    val outputTensors = torch.hsplit(inputTensor, indicesOrSections = 2)
    assert(
      outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
        outputTensor.allclose(expectedTensor)
      }
    )
  }

  testUnaryOp(
    op = (t: Tensor[?]) => hstack(Seq(t, t)),
    opName = "hstack",
    inputTensor = Tensor(Seq(1, 2, 3)),
    expectedTensor = Tensor(Seq(1, 2, 3, 1, 2, 3))
  )

  val index = Tensor(Seq(0L, 4L, 2L))
  val source = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9)))
  propertyTestUnaryOp(
    op = (t: Tensor[?]) => indexAdd(_, 0, index, source.to(dtype = t.dtype)),
    opName = "indexAdd"
  )
  test("indexAdd.unit-test") {
    val inputTensor = torch.ones(Seq(5, 3))
    val expectedTensor = Tensor(
      Seq(
        Seq(2.0, 3.0, 4.0),
        Seq(1.0, 1.0, 1.0),
        Seq(8.0, 9.0, 10.0),
        Seq(1.0, 1.0, 1.0),
        Seq(5.0, 6.0, 7.0)
      )
    ).to(dtype = torch.float32)
    val output = torch.indexAdd(
      inputTensor,
      dim = 0,
      index = index,
      source = source.to(dtype = torch.float32)
    )
    assert(output.allclose(expectedTensor))
  }

  propertyTestUnaryOp(
    op = (t: Tensor[?]) =>
      indexCopy(
        t,
        0,
        torch.arange(end = 4).to(dtype = int64),
        torch.ones(Seq(4, 4)).to(dtype = t.dtype)
      ),
    opName = "indexCopy"
  )
  test("indexCopy.unit-test") {
    val inputTensor = torch.zeros(Seq(5, 3))
    val source =
      torch.Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9))).to(dtype = torch.float32)
    val index = torch.Tensor(Seq(0L, 4L, 2L))
    val expectedTensor = Tensor(
      Seq(
        Seq(1.0, 2.0, 3.0),
        Seq(0.0, 0.0, 0.0),
        Seq(7.0, 8.0, 9.0),
        Seq(0.0, 0.0, 0.0),
        Seq(4.0, 5.0, 6.0)
      )
    ).to(dtype = torch.float32)
    val output = torch.indexCopy(inputTensor, 0, index, source)
    assert(output.allclose(expectedTensor))
  }

  // TODO testUnaryOp(op = indexReduce, opName = "indexReduce", inputTensor = ???, expectedTensor = ???)

  testUnaryOp(
    op = (t: Tensor[?]) => {
      val index = torch.Tensor(Seq(0L, 2L))
      indexSelect(t, 0, index)
    },
    opName = "indexSelect",
    inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9))),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3), Seq(7, 8, 9)))
  )

  val mask = Tensor(
    Seq(
      Seq(false, false, false, false),
      Seq(false, true, true, true),
      Seq(false, false, false, true),
      Seq(false, false, false, false)
    )
  )
  testUnaryOp(
    op = maskedSelect(_, mask),
    opName = "maskedSelect",
    inputTensor =
      Tensor(Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(9, 10, 11, 12), Seq(13, 14, 15, 16))),
    expectedTensor = Tensor(Seq(6, 7, 8, 12))
  )

  testUnaryOp(
    op = movedim(_, 0, 1),
    opName = "movedim",
    inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6))),
    expectedTensor = Tensor(Seq(Seq(1, 4), Seq(2, 5), Seq(3, 6)))
  )

  testUnaryOp(
    op = moveaxis(_, 0, 1),
    opName = "moveaxis",
    inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6))),
    expectedTensor = Tensor(Seq(Seq(1, 4), Seq(2, 5), Seq(3, 6)))
  )

  testUnaryOp(
    op = narrow(_, 0, 0, 2),
    opName = "narrow",
    inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9))),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
  )

  testUnaryOp(
    op = narrowCopy(_, 0, 0, 2),
    opName = "narrowCopy",
    inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9))),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
  )

  testUnaryOp(
    op = nonzero,
    opName = "nonzero",
    inputTensor = Tensor(Seq(1, 1, 1, 0, 1)),
    expectedTensor = Tensor(Seq(Seq(0L), Seq(1L), Seq(2L), Seq(4L)))
  )

  testUnaryOp(
    op = permute(_, 1, 0),
    opName = "permute",
    inputTensor = Tensor(Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8))),
    expectedTensor = Tensor(Seq(Seq(1, 5), Seq(2, 6), Seq(3, 7), Seq(4, 8)))
  )

  testUnaryOp(
    op = reshape(_, 1, 16),
    opName = "reshape",
    inputTensor =
      Tensor(Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(9, 10, 11, 12), Seq(13, 14, 15, 16))),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)))
  )

  testUnaryOp(
    op = select(_, 0, 1),
    opName = "select",
    inputTensor = Tensor(Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8))),
    expectedTensor = Tensor(Seq(Seq(5, 6, 7, 8)))
  )

  // TODO testUnaryOp(op = scatter, opName = "scatter", inputTensor = ???, expectedTensor = ???)
  // TODO testUnaryOp(op = diagonalScatter, opName = "diagonalScatter", inputTensor = ???, expectedTensor = ???)
  // TODO testUnaryOp(op = selectScatter, opName = "selectScatter", inputTensor = ???, expectedTensor = ???)
  // TODO testUnaryOp(op = sliceScatter, opName = "sliceScatter", inputTensor = ???, expectedTensor = ???)
  // TODO testUnaryOp(op = scatterAdd, opName = "scatterAdd", inputTensor = ???, expectedTensor = ???)
  // TODO testUnaryOp(op = scatterReduce, opName = "scatterReduce", inputTensor = ???, expectedTensor = ???)

  propertyTestUnaryOp(op = split(_, 2), opName = "split")
  test("split.unit-test") {
    val inputTensor = torch.arange(end = 10).reshape(5, 2)
    val expectedTensors = Seq(
      Tensor(Seq(Seq(0, 1), Seq(2, 3))),
      Tensor(Seq(Seq(4, 5), Seq(6, 7))),
      Tensor(Seq(Seq(8, 9)))
    )
    val outputTensors = torch.split(inputTensor, splitSizeOrSections = 2)
    outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
      outputTensor.allclose(expectedTensor)
    }
  }

  testUnaryOp(
    op = squeeze(_),
    opName = "squeeze",
    inputTensor = Tensor(Seq(Seq(Seq(2, 2)), Seq(Seq(3, 3)))),
    expectedTensor = Tensor(Seq(Seq(2, 2), Seq(3, 3)))
  )

  testUnaryOp(
    op = (t: Tensor[?]) => stack(Seq(t, t), 0),
    opName = "stack",
    inputTensor = Tensor(Seq(1, 2, 3)),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3), Seq(1, 2, 3)))
  )

  testUnaryOp(
    op = swapaxes(_, 0, 1),
    opName = "swapaxes",
    inputTensor = Tensor(Seq(Seq(Seq(0, 1), Seq(2, 3)), Seq(Seq(4, 5), Seq(6, 7)))),
    expectedTensor = Tensor(Seq(Seq(Seq(0, 1), Seq(4, 5)), Seq(Seq(2, 3), Seq(6, 7))))
  )

  testUnaryOp(
    op = swapdims(_, 0, 1),
    opName = "swapdims",
    inputTensor = Tensor(Seq(Seq(Seq(0, 1), Seq(2, 3)), Seq(Seq(4, 5), Seq(6, 7)))),
    expectedTensor = Tensor(Seq(Seq(Seq(0, 1), Seq(4, 5)), Seq(Seq(2, 3), Seq(6, 7))))
  )

  testUnaryOp(
    op = t,
    opName = "t",
    inputTensor = Tensor(Seq(Seq(0, 1, 2), Seq(3, 4, 5))),
    expectedTensor = Tensor(Seq(Seq(0, 3), Seq(1, 4), Seq(2, 5)))
  )

  testUnaryOp(
    op = {
      val index = torch.Tensor(Seq(0L, 2L, 5L))
      take(_, index)
    },
    opName = "take",
    inputTensor = Tensor(Seq(Seq(4, 3, 5), Seq(6, 7, 8))),
    expectedTensor = Tensor(Seq(4, 5, 8))
  )

  testUnaryOp(
    op = {
      val index = Tensor(Seq(3L))
      takeAlongDim(_, index, None)
    },
    opName = "takeAlongDim",
    inputTensor = Tensor(Seq(Seq(10, 20, 30), Seq(60, 40, 50))),
    expectedTensor = Tensor(Seq(60))
  )

  propertyTestUnaryOp(op = tensorSplit(_, 1, 0), opName = "tensorSplit")
  test("tensorSplit.unit-test") {
    val inputTensor = torch.arange(end = 8)
    val expectedTensors = Seq(
      Tensor(Seq(0, 1, 2)),
      Tensor(Seq(3, 4, 5)),
      Tensor(Seq(6, 7))
    )
    val outputTensors = torch.tensorSplit(inputTensor, 3)
    assert(
      outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
        outputTensor.allclose(expectedTensor)
      }
    )
  }

  testUnaryOp(
    op = tile(_, 2),
    opName = "tile",
    inputTensor = Tensor(Seq(1, 2, 3)),
    expectedTensor = Tensor(Seq(1, 2, 3, 1, 2, 3))
  )

  testUnaryOp(
    op = transpose(_, 0, 1),
    opName = "transpose",
    inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6))),
    expectedTensor = Tensor(Seq(Seq(1, 4), Seq(2, 5), Seq(3, 6)))
  )

  propertyTestUnaryOp(op = unbind(_, 0), opName = "unbind")
  test("unbind.unit-test") {
    val inputTensor = Tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    val expectedTensors = Seq(Tensor(Seq(1, 2, 3)), Tensor(Seq(4, 5, 6)))
    val outputTensors = torch.unbind(inputTensor)
    assert(
      outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
        outputTensor.allclose(expectedTensor)
      }
    )
  }

  testUnaryOp(
    op = unsqueeze(_, 1),
    opName = "unsqueeze",
    inputTensor = Tensor(Seq(1, 2, 3, 4)),
    expectedTensor = Tensor(Seq(Seq(1), Seq(2), Seq(3), Seq(4)))
  )

  propertyTestUnaryOp(op = vsplit(_, 2), opName = "vsplit")
  test("vsplit.unit-test") {
    val inputTensor = torch.arange(end = 16.0).reshape(4, 4).to(dtype = torch.float64)
    val expectedTensors = Seq(
      Tensor(
        Seq(
          Seq(0.0, 1.0, 2.0, 3.0),
          Seq(4.0, 5.0, 6.0, 7.0)
        )
      ),
      Tensor(
        Seq(
          Seq(8.0, 9.0, 10.0, 11.0),
          Seq(12.0, 13.0, 14.0, 15.0)
        )
      )
    )
    val outputTensors = torch.vsplit(inputTensor, 2)
    assert(
      outputTensors.zip(expectedTensors).forall { case (outputTensor, expectedTensor) =>
        outputTensor.allclose(expectedTensor)
      }
    )
  }

  testUnaryOp(
    op = (t: Tensor[?]) => vstack(Seq(t, t)),
    opName = "vstack",
    inputTensor = Tensor(Seq(1, 2, 3)),
    expectedTensor = Tensor(Seq(Seq(1, 2, 3), Seq(1, 2, 3)))
  )

  propertyTestUnaryOp(
    op = (t: Tensor[?]) => {
      val conditionWhere =
        Tensor(
          Seq(
            Seq(true, true, false, false),
            Seq(true, true, false, false),
            Seq(true, true, false, false),
            Seq(true, true, false, false)
          )
        )
      where(conditionWhere, t, t)
    },
    opName = "where"
  )
  test("where.unit-test") {
    val condition = Tensor(Seq(Seq(false, true), Seq(true, false)))
    val input = Tensor(Seq(Seq(1, 2), Seq(3, 4)))
    val other = Tensor(Seq(Seq(-1, -2), Seq(-3, -4)))
    val exptectedOutput = Tensor(Seq(Seq(-1, 2), Seq(3, -4)))
    val output = torch.where(condition, input, other)
    assert(output.allclose(exptectedOutput))
  }
}
