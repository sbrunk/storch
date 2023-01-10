package torch
package nn
package modules
package pooling

class MaxPoolSuite extends munit.FunSuite {
  test("MaxPool2d output shapes") {
    val input  = torch.randn(Seq(1,3,244,244))
    // pool of square window of size=3, stride=2
    val m1     = MaxPool2d[Float32](3, stride=Some(2))
    assertEquals(m1(input).shape, Seq[Long](1, 3, 121, 121))
    // pool of non-square window
    val m2 = MaxPool2d[Float32]((3, 2), stride=Some(2, 1))
    assertEquals(m2(input).shape, Seq[Long](1, 3, 121, 243))
    val m3 = MaxPool2d[Float32](3)
    assertEquals(m3(input).shape, Seq[Long](1, 3, 81, 81))
  }
}
