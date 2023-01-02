package torch
package nn

import org.bytedeco.pytorch
import org.bytedeco.pytorch.ReLUImpl
import org.bytedeco.pytorch.ReLUOptions

/** Applies the rectified linear unit function element-wise:
  *
  * $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final case class ReLU(inplace: Boolean = false) extends Module:
  private val options = new ReLUOptions()
  options.inplace().put(inplace)

  override protected[torch] val nativeModule: ReLUImpl = ReLUImpl()

  override def registerWithParent[M <: pytorch.Module](parent: M)(using name: sourcecode.Name): Unit =
    // println(s"registering ${name.value}: $this with $parent")
    parent.register_module(name.value, nativeModule)

  def apply[D <: DType](t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))
