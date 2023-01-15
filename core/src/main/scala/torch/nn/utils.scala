package torch
package nn

import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.TensorVector

object utils:

    def clipGradNorm_(parameters: Seq[Tensor[?]], max_norm: Double, norm_type: Double = 2.0, error_if_nonfinite: Boolean = false) =
    torchNative.clip_grad_norm_(TensorVector(parameters.map(_.native).toArray*), max_norm, norm_type, error_if_nonfinite)

