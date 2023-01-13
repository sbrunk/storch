package torch

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative

/** A memoryFormat is an object representing the memory format on which a torch.Tensor is or will be allocated. */
enum MemoryFormat:
  /** Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order.
    */
  case Contiguous
  /** Used in functions like clone to preserve the memory format of the input tensor. If input tensor is allocated in
   * dense non-overlapping memory, the output tensor strides will be copied from the input.
   * Otherwise output strides will follow torch.contiguous_format */
  case Preserve
  /** Tensor is or will be allocated in dense non-overlapping memory.
    * Strides represented by values in `strides[0] > strides[2] > strides[3] > strides[1] == 1` aka NHWC order.
    */
  case ChannelsLast
  case ChannelsLast3d

  private[torch] def toNative: torchNative.MemoryFormat = torchNative.MemoryFormat.valueOf(this.toString)
  private[torch] def toNativeOptional: pytorch.MemoryFormatOptional = pytorch.MemoryFormatOptional(torchNative.MemoryFormat.valueOf(this.toString))


