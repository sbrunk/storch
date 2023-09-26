
import torch.*

/**
 * Base class for model's outputs that also contains a pooling of the last hidden states.
 *
 * @constructor create a new instance of BaseModelOutputWithPooling
 * @param last_hidden_state Sequence of hidden-states at the output of the last layer of the model.
 *                          Tensor of shape `(batch_size, sequence_length, hidden_size)`.
 * @param pooler_output Last layer hidden-state of the first token of the sequence (classification token) after further processing
 *                      through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
 *                      the classification token after processing through a linear layer and a tanh activation function. The linear
 *                      layer weights are trained from the next sentence prediction (classification) objective during pretraining.
 *                      Tensor of shape `(batch_size, hidden_size)`.
 * @param hidden_states Tuple of `Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
 *                      one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
 *                      Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
 *                      This param is optional and returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`.
 * @param attentions Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
 *                   Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
 *                   This param is optional and returned when `output_attentions=True` is passed or when `config.output_attentions=True`.
 */

case class BaseModelOutputWithPooling(
    lastHiddenState: Tensor[Float32],
    poolerOutput: Option[Tensor[Float32]] = None,
    hiddenStates: Seq[Tensor[Float32]] = Seq.empty,
    attentions: Seq[Tensor[Float32]] = Seq.empty
)

import scala.Option
import scala.Tuple2

/**
  * Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
  *
  * @param lastHiddenState Sequence of hidden-states at the output of the last layer of the model. 
  *                        FloatTensor of shape (batch_size, sequence_length, hidden_size)
  *                        If `pastKeyValues` is used only the last hidden-state of the sequences of shape 
  *                        (batch_size, 1, hidden_size) is output.
  * @param pastKeyValues   Tuple of `Tuple2(FloatTensor, FloatTensor)` of length `config.nLayers`, with each tuple 
  *                        having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head). 
  *                        Intended to be returned when `use_cache=True` is passed or when `config.useCache=True`.
  *                        Contains pre-computed hidden-states (key and values in the self-attention blocks and 
  *                        optionally if `config.isEncoderDecoder=True` in the cross-attention blocks) that can be 
  *                        used (see `pastKeyValues` input) to speed up sequential decoding.
  * @param hiddenStates    Tuple of FloatTensor (one for the output of the embeddings, if the model has an embedding 
  *                        layer, + one for the output of each layer) of shape `(batch_size, sequence_length, 
  *                        hidden_size)`. Rolled out when `output_hidden_states=True` is passed or when 
  *                        `config.outputHiddenStates=True`.
  *                        Hidden-states of the model at the output of each layer plus the optional initial embedding 
  *                        outputs.
  * @param attentions      Tuple of FloatTensor (one for each layer) of shape `(batch_size, num_heads, 
  *                        sequence_length, sequence_length)`.
  *                        Returned when `output_attentions=True` is passed or when `config.outputAttentions=True`.
  *                        Attentions weights after the attention softmax, used to compute the weighted average in the 
  *                        self-attention heads.
  */
case class BaseModelOutputWithPast(
    lastHiddenState: Tensor[Float32],
    pastKeyValues: Option[Seq[(Tensor[Float32], Tensor[Float32])]] = None,
    hiddenStates: Option[Seq[Tensor[Float32]]] = None,
    attentions: Option[Seq[Tensor[Float32]]] = None
)