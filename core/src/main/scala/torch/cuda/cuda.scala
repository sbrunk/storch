package torch
package cuda

import org.bytedeco.pytorch.global.torch as torchNative

/** Returns a Boolean indicating if CUDA is currently available. */
def isAvailable: Boolean = torchNative.cuda_is_available()
