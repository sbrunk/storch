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

object indexing:

  case class Slice(start: Option[Int], end: Option[Int], step: Option[Int])
  object Slice:
    private def extract(index: Option[Int] | Int) = index match
      case i: Option[Int] => i
      case i: Int         => Option(i)
    def apply(
        start: Option[Int] | Int = None,
        end: Option[Int] | Int = None,
        step: Option[Int] | Int = None
    ): Slice = Slice(extract(start), extract(end), extract(step))

  /** Ellipsis or ... in Python syntax. */
  sealed class Ellipsis

  /** Ellipsis or ... in Python syntax. */
  case object Ellipsis extends Ellipsis

  /** Ellipsis or ... in Python syntax. */
  val --- = Ellipsis

  /** Range (colon / :) in python syntax. */
  val :: = Slice()

  /** Allow for {{{t(1.::)}}} and {{{t(1.::(2)}}} */
  extension (start: Int | Option[Int])
    def ::(step: Int | Option[Int]): Slice =
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and start are reversed here
      Slice(step, None, start)

    def :: : Slice = Slice(start, None, None)

export indexing.*
