package torch
/** 
  * 
  * @groupname nn_conv Convolution Layers
  * @groupname nn_linear Linear Layers
  * 
  */
package object nn {
  export modules.Module
  export modules.Default

  export modules.activation.Softmax
  export modules.activation.ReLU
  export modules.batchnorm.BatchNorm2d
  export modules.container.Sequential
  export modules.conv.Conv2d
  export modules.linear.Linear
  export modules.linear.Identity
  export modules.normalization.GroupNorm
  export modules.pooling.AdaptiveAvgPool2d
  export modules.pooling.MaxPool2d
}
