package torch
package nn
package modules
package pooling

class AdapativeAvgPool2dSuite extends munit.FunSuite {
  test("AdapativeAvgPool2d output shapes") {
    val m1     = AdaptiveAvgPool2d((5, 7))
    val input  = torch.randn(Seq(1, 64, 8, 9))
    assertEquals(m1(input).shape, Seq[Long](1, 64, 5, 7))
    val m2 = nn.AdaptiveAvgPool2d((1, 1))
    assertEquals(m2(input).shape, Seq[Long](1, 64, 1, 1))
  }
}
