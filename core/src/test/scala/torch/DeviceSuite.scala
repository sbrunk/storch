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
import org.scalacheck.Prop.*
import org.scalacheck._
import Generators.given

class DeviceSuite extends ScalaCheckSuite {
  test("device native roundtrip") {
    val d = Device("cpu")
    assertEquals(d, Device(d.toNative))
  }

  property("device native roundtrip for all") {
    forAll { (d: Device) =>
      assertEquals(d, Device(d.toNative))
    }
  }
}
