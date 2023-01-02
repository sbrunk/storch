package torch
package nn

import org.bytedeco.pytorch.IdentityImpl

/** A placeholder identity operator that is argument-insensitive.
 *
 *  @group nn_linear
 *
 */
class Identity(args: Any*) extends Module:
  override val nativeModule: IdentityImpl = IdentityImpl()

  def forward[D <: DType](t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))
