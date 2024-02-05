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
package optim

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{SGDOptions, TensorVector}

import scala.collection.immutable.Iterable

// format: off
/** Implements stochastic gradient descent (optionally with momentum).
 *
 * $$
 * \begin{aligned}
 *   &\rule{110mm}{0.4pt}                                                                 \\
 *   &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
 *       \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
 *   &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
 *   \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
 *   &\rule{110mm}{0.4pt}                                                                 \\
 *   &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
 *   &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
 *   &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
 *   &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
 *   &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
 *   &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
 *   &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
 *   &\hspace{10mm}\textbf{else}                                                          \\
 *   &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
 *   &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
 *   &\hspace{15mm} g_t \leftarrow g_{t-1} + \mu \textbf{b}_t                             \\
 *   &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
 *   &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
 *   &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
 *   &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
 *   &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
 *   &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
 *   &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
 *   &\bf{return} \:  \theta_t                                                     \\[-1.ex]
 *   &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
 * \end{aligned}
 *  $$
 *
 *  Nesterov momentum is based on the formula from
 *  [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)
 */
// format: on
// TODO optionial parameters
class SGD(params: Iterable[Tensor[?]], lr: Float) extends Optimizer {
  private val nativeParams = TensorVector(params.map(_.native).toArray*)
  private val options = SGDOptions(lr)
  override private[torch] val native: pytorch.SGD = pytorch.SGD(nativeParams, options)
}
