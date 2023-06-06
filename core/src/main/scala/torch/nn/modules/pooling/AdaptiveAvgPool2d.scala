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
package pooling

import org.bytedeco.pytorch.AdaptiveAvgPool2dImpl
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch

import torch.internal.NativeConverters.{toNative, toOptional}
import org.bytedeco.pytorch.LongOptionalVector
import org.bytedeco.pytorch.LongOptional
import scala.annotation.nowarn

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class AdaptiveAvgPool2d(
    outputSize: Int | Option[Int] | (Option[Int], Option[Int]) | (Int, Int)
) extends Module {

  // TODO find a better way to remove that warning instead of just silencing
  // See https://users.scala-lang.org/t/alternative-for-type-ascriptions-after-tuple-patterns-in-scala-3-3-0/9328
  @nowarn("msg=Type ascriptions after patterns other than")
  private def nativeOutputSize = outputSize match
    case (h: Int, w: Int) => new LongOptionalVector(new LongOptional(h), new LongOptional(w))
    case x: Int           => new LongOptionalVector(new LongOptional(x), new LongOptional(x))
    case (h, w): (Option[Int], Option[Int]) =>
      new LongOptionalVector(toOptional(h.map(_.toLong)), toOptional(w.map(_.toLong)))
    case x: Option[Int] =>
      new LongOptionalVector(toOptional(x.map(_.toLong)), toOptional(x.map(_.toLong)))

  override private[torch] val nativeModule: AdaptiveAvgPool2dImpl = AdaptiveAvgPool2dImpl(
    nativeOutputSize.get(0)
  )

  override def registerWithParent[T <: pytorch.Module](parent: T)(using
      name: sourcecode.Name
  ): Unit =
    parent.register_module(name.value, nativeModule)

  def apply[D <: BFloat16 | Float32 | Float64](t: Tensor[D]): Tensor[D] = Tensor(
    nativeModule.forward(t.native)
  )
}
