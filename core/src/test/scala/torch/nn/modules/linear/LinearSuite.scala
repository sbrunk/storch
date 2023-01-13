package torch
package nn
package modules
package linear

class LinearSuite extends munit.FunSuite {
  test("Linear shape") {
    val linear     = Linear(20,30)
    val input = randn(Seq(128, 20))
    assertEquals(linear(input).shape, Seq[Long](128, 30))
  }
}
