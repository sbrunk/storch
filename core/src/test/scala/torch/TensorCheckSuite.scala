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

import munit.ScalaCheckSuite
import shapeless3.typeable.{TypeCase, Typeable}
import shapeless3.typeable.syntax.typeable.*
import Generators.*
import org.scalacheck.Prop.*

import scala.util.Try
import org.scalacheck.Gen

trait TensorCheckSuite extends ScalaCheckSuite {

  given tensorTypeable[T <: DType](using tt: Typeable[T]): Typeable[Tensor[T]] with
    def castable(t: Any): Boolean =
      t match
        case (tensor: Tensor[?]) =>
          tensor.dtype.castable[T]
        case _ => false
    def describe = s"Tensor[${tt.describe}]"

  private def propertyTestName(opName: String) = s"${opName}.property-test"
  private def unitTestName(opName: String) = s"${opName}.unit-test"

  inline def propertyTestBinaryOp[InA <: DType, InB <: DType](
      op: Function2[Tensor[InA], Tensor[InB], ?],
      opName: String,
      skipPropertyTestReason: Option[String] = None
  ): Unit =
    property(propertyTestName(opName)) {
      assume(skipPropertyTestReason.isEmpty, skipPropertyTestReason)

      // TODO Validate output types
      val tensorInACase = TypeCase[Tensor[InA]]
      val tensorInBCase = TypeCase[Tensor[InB]]
      forAll(genTensor(), genTensor()) {
        case (tensorInACase(tensorA), tensorInBCase(tensorB)) =>
          val result = Try(op(tensorA, tensorB))
          assert(
            result.isSuccess,
            s"""|
                |Tensor operation 'torch.${opName}' does not support (${tensorA.dtype}, ${tensorB.dtype}) inputs
                |
                |${result.failed.get}
                """.stripMargin
          )
        case (tensorA, tensorB) =>
          val result = Try(op(tensorA.asInstanceOf[Tensor[InA]], tensorB.asInstanceOf[Tensor[InB]]))
          assert(
            result.isFailure,
            s"""|
                |Tensor operation 'torch.${opName}' supports (A: ${tensorA.dtype}, B: ${tensorB.dtype}) inputs but storch interface is currently restricted to
                |Type A: ${tensorInACase}
                |Type B: ${tensorInBCase}
                """.stripMargin
          )
      }
    }

  inline def propertyTestUnaryOp[In <: DType](
      op: Function1[Tensor[In], ?],
      opName: String,
      inline genTensor: Gen[Tensor[In]] = genTensor[In]()
  ): Unit =
    property(propertyTestName(opName)) {
      // TODO Validate output types
      val tensorInCase = TypeCase[Tensor[In]]
      forAll(genTensor) {
        case tensorInCase(tensor) =>
          val result = Try(op(tensor))
          assert(
            result.isSuccess,
            s"""|
                |Tensor operation 'torch.${opName}' does not support ${tensor.dtype} inputs
                |
                |${result.failed.get}
                """.stripMargin
          )
        case tensor =>
          val result = Try(op(tensor.asInstanceOf[Tensor[In]]))
          assert(
            result.isFailure,
            s"""|
                |Tensor operation 'torch.${opName}' supports ${tensor.dtype} inputs but storch interface is currently restricted to ${tensorInCase}
                """.stripMargin
          )
      }
    }

  inline def unitTestBinaryOp[
      InA <: DType,
      InAS <: ScalaType,
      InB <: DType,
      InBS <: ScalaType
  ](
      op: Function2[Tensor[InA], Tensor[InB], Tensor[?]],
      opName: String,
      inline inputTensors: (Tensor[ScalaToDType[InAS]], Tensor[ScalaToDType[InBS]]),
      inline expectedTensor: Tensor[?],
      absolutePrecision: Double = 1e-04
  )(using ScalaToDType[InAS] <:< InA, ScalaToDType[InBS] <:< InB): Unit =
    test(unitTestName(opName)) {
      val outputTensor = op(
        inputTensors._1.asInstanceOf[Tensor[InA]],
        inputTensors._2.asInstanceOf[Tensor[InB]]
      )
      val allclose = outputTensor.allclose(
        other = expectedTensor,
        atol = absolutePrecision,
        equalNan = true
      )
      assert(
        allclose,
        s"""|
            |Tensor results are not all close for 'torch.${opName}'
            |
            |Input tensors:
            |${inputTensors}
            |
            |Output tensor:
            |${outputTensor}
            |
            |Expected tensor:
            |${expectedTensor}""".stripMargin
      )
    }

  inline def unitTestUnaryOp[In <: DType, InS <: ScalaType](
      op: Function1[Tensor[In], Tensor[?]],
      opName: String,
      inline inputTensor: Tensor[ScalaToDType[InS]],
      inline expectedTensor: Tensor[?],
      absolutePrecision: Double = 1e-04
  )(using ScalaToDType[InS] <:< In): Unit =
    test(unitTestName(opName)) {
      val outputTensor = op(inputTensor.asInstanceOf[Tensor[In]])
      val allclose = outputTensor.allclose(
        other = expectedTensor,
        atol = absolutePrecision,
        equalNan = true
      )
      assert(
        allclose,
        s"""|
            |Tensor results are not all close for 'torch.${opName}'
            |
            |Input tensor:
            |${inputTensor}
            |
            |Output tensor:
            |${outputTensor}
            |
            |Expected tensor:
            |${expectedTensor}""".stripMargin
      )
    }

  inline def testBinaryOp[InA <: DType, InAS <: ScalaType, InB <: DType, InBS <: ScalaType](
      op: Function2[Tensor[InA], Tensor[InB], Tensor[?]],
      opName: String,
      inline inputTensors: (Tensor[ScalaToDType[InAS]], Tensor[ScalaToDType[InBS]]),
      inline expectedTensor: Tensor[?],
      absolutePrecision: Double = 1e-04,
      skipPropertyTestReason: Option[String] = None
  )(using ScalaToDType[InAS] <:< InA, ScalaToDType[InBS] <:< InB): Unit =
    propertyTestBinaryOp(op, opName, skipPropertyTestReason)
    unitTestBinaryOp(op, opName, inputTensors, expectedTensor, absolutePrecision)

  inline def testUnaryOp[In <: DType, InS <: ScalaType](
      op: Function1[Tensor[In], Tensor[?]],
      opName: String,
      inline inputTensor: Tensor[ScalaToDType[InS]],
      inline expectedTensor: Tensor[?],
      absolutePrecision: Double = 1e-04,
      inline genTensor: Gen[Tensor[In]] = genTensor[In]()
  )(using ScalaToDType[InS] <:< In): Unit =
    propertyTestUnaryOp(op, opName, genTensor)
    unitTestUnaryOp(op, opName, inputTensor, expectedTensor, absolutePrecision)

}
