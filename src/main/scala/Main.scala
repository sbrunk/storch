import torch.*

@main
def main(): Unit = {
  /* try to use MKL when available */
  System.setProperty("org.bytedeco.openblas.load", "mkl")
  val t = Tensor(Seq(1,2,3), shape = 3)
  println(eye(3) * 2 * t)
  println(ones(2))
}