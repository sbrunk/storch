package torch
/** 
  * 
  * @groupname nn_conv Convolution Layers
  * @groupname nn_linear Linear Layers
  * 
  */
package object nn {
  type AdaptiveAvgPool2d = modules.pooling.AdaptiveAvgPool2d
  val AdaptiveAvgPool2d = modules.pooling.AdaptiveAvgPool2d
  type GroupNorm[ParamType <: DType] = modules.GroupNorm[ParamType]
  val GroupNorm = modules.GroupNorm

  type Sequential[D <: DType] = modules.container.Sequential[D]
  val Sequential = modules.container.Sequential
}
