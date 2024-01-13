/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch

/** These are the basic building blocks for graphs.
  *
  * @groupname nn_conv Convolution Layers
  * @groupname nn_linear Linear Layers
  * @groupname nn_utilities Utilities
  */
package object nn {

  export modules.Module

  export modules.activation.Softmax
  export modules.activation.LogSoftmax
  export modules.activation.ReLU
  export modules.activation.Tanh
  export modules.batchnorm.BatchNorm1d
  export modules.batchnorm.BatchNorm2d
  export modules.container.Sequential
  export modules.container.ModuleList
  export modules.conv.Conv2d
  export modules.flatten.Flatten
  export modules.linear.Linear
  export modules.linear.Identity
  export modules.normalization.GroupNorm
  export modules.normalization.LayerNorm
  export modules.pooling.AdaptiveAvgPool2d
  export modules.pooling.MaxPool2d
  export modules.sparse.Embedding
  export modules.regularization.Dropout

  export loss.CrossEntropyLoss
}
