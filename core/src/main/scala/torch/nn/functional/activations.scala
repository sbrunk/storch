package torch
package nn
package functional

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.javacpp.LongPointer
import torch.internal.NativeConverters.toOptional
import org.bytedeco.pytorch.{ScalarTypeOptional, TensorOptional}

/** Applies a softmax followed by a logarithm.
  *
  * While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically
  * unstable. This function uses an alternative formulation to compute the output and gradient correctly.
  *
  * See `torch.nn.LogSoftmax` for more details.
  */
def logSoftmax[In <: DType, Out <: DType](input: Tensor[In], dim: Long)(dtype: Out = input.dtype): Tensor[Out] =
  val nativeDType = if dtype == input.dtype then ScalarTypeOptional() else ScalarTypeOptional(dtype.toScalarType)
  Tensor(torchNative.log_softmax(input.native, dim, nativeDType))

  /** Applies the rectified linear unit function element-wise.
    *
    * See `torch.nn.ReLU` for more details.
    */
def sigmoid[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.sigmoid(input.native))

/** Applies the element-wise function $\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$
  *
  * See `torch.nn.Sigmoid` for more details.
  */
def relu[D <: DType](input: Tensor[D]): Tensor[D] = Tensor(torchNative.relu(input.native))

/** Applies a softmax function. */
def softmax[In <: DType, Out <: DType](input: Tensor[In], dim: Long)(dtype: Out = input.dtype): Tensor[Out] =
  val nativeDType = if dtype == input.dtype then ScalarTypeOptional() else ScalarTypeOptional(dtype.toScalarType)
  Tensor(torchNative.softmax(input.native, dim, nativeDType))
