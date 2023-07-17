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

import shapeless3.typeable.{TypeCase, Typeable}
import shapeless3.typeable.syntax.typeable.*
import scala.util.NotGiven
import spire.math.Complex

/* Typeable instance for Array[T]
 * NOTE: It needs to iterate through the whole array to validate casteability
 */
given iterableTypeable[T](using tt: Typeable[T]): Typeable[Array[T]] with
  def castable(t: Any): Boolean =
    t match
      case (arr: Array[?]) =>
        arr.forall(_.castable[T])
      case _ => false
  def describe = s"Array[${tt.describe}]"

/* TypeCase helpers to perform pattern matching on `Complex` higher kinded types */
val complexDoubleArray = TypeCase[Array[Complex[Double]]]
val complexFloatArray = TypeCase[Array[Complex[Float]]]

/* TypeCase helpers to perform pattern matching on `Seq` higher kinded types */
val singleSeq = TypeCase[Seq[?]]
val doubleSeq = TypeCase[Seq[Seq[?]]]
val tripleSeq = TypeCase[Seq[Seq[Seq[?]]]]

/* Type helper to describe inputs that accept Tensor or Real scalars */
type TensorOrReal[D <: RealNN] = Tensor[D] | Real

/* Evidence used in operations where Bool is accepted, but only on one of the two inputs, not both
 */
type OnlyOneBool[A <: DType, B <: DType] = NotGiven[A =:= Bool & B =:= Bool]

/* Evidence used in operations where at least one Float is required */
type AtLeastOneFloat[A <: DType, B <: DType] = A <:< FloatNN | B <:< FloatNN

/* Evidence used in operations where at least one Float or Complex is required */
type AtLeastOneFloatOrComplex[A <: DType, B <: DType] = A <:< (FloatNN | ComplexNN) |
  B <:< (FloatNN | ComplexNN)

/* Evidence that two dtypes are not the same */
type NotEqual[D <: DType, D2 <: DType] = NotGiven[D =:= D2]
