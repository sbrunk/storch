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

case class Slice(start: Option[Long], end: Option[Long], step: Option[Long])
object Slice:
  private def extract(index: Option[Long] | Long) = index match
    case i: Option[Long] => i
    case i: Long         => Option(i)
  def apply(
    start: Option[Long] | Long = None,
    end: Option[Long] | Long = None,
    step: Option[Long] | Long = None
  ): Slice = Slice(extract(start), extract(end), extract(step))