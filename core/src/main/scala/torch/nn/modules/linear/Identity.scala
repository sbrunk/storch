package torch
package nn
package modules
package linear

import org.bytedeco.pytorch.IdentityImpl
import torch.nn.modules.Module
import torch.{DType, Tensor}

/** A placeholder identity operator that is argument-insensitive.
 *
 *  @group nn_linear
 *
 */
class Identity(args: Any*) extends Module:
  override val nativeModule: IdentityImpl = IdentityImpl()

  def forward[D <: DType](t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))
