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

import torch.*
import torch.nn.functional as F
import RopeScaling.Type

case class RopeScaling(
    tpe: RopeScaling.Type,
    factor: Float
) {
  assert(factor > 1f, s"rope_scaling`'s factor field must be an float > 1, got $factor")
}

object RopeScaling:
  enum Type:
    case Linear, Dynamic

case class PretrainedConfig(
  output_hidden_states: Boolean = false,
  output_attentions: Boolean = false
)

case class LlamaConfig(
    vocab_size: Int = 32000,
    hidden_size: Int = 4096,
    intermediate_size: Int = 11008,
    num_hidden_layers: Int = 32,
    num_attention_heads: Int = 32,
    num_key_value_heads: Int,
    hidden_act: String = "silu", // TODO
    max_position_embeddings: Int = 2048,
    initializer_range: Float = 0.02,
    rms_norm_eps: Float = 1e-6,
    use_cache: Boolean = true,
    pad_token_id: Int = 0,
    bos_token_id: Int = 1,
    eos_token_id: Int = 2,
    pretraining_tp: Int = 1,
    tie_word_embeddings: Boolean = false,
    rope_scaling: Option[RopeScaling] = None,
    pretrainedCondig: PretrainedConfig = PretrainedConfig()
) {
  export pretrainedCondig.*
}

object Llama {

  // Copied from transformers.models.bart.modeling_bart._make_causal_mask
  /**
      Make causal mask used for bi-directional self-attention.
    */
  def _make_causal_mask[D <: FloatNN](
      input_ids_shape: Seq[Int], dtype: D, device: Device, past_key_values_length: Int = 0) =
    val Seq(bsz, tgt_len) = input_ids_shape
    var mask =
      val mask = torch.full(Seq(tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
      val mask_cond = torch.arange(end=mask.size(-1), device=device)
      mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
      mask.to(dtype)
    if past_key_values_length > 0 then
      mask = torch.cat(Seq(torch.zeros(Seq(tgt_len, past_key_values_length), dtype = dtype, device=device), mask), dim = -1)
    mask(None, None, ::, ::).expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


  // Copied from transformers.models.bart.modeling_bart._expand_mask
  /** 
      Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    */
  def _expand_mask[D <: FloatNN](mask: Tensor[Bool], dtype: D, tgt_len: Option[Int] = None) =
      val Seq(bsz, src_len) = mask.size
      val _tgt_len: Int = tgt_len.getOrElse(src_len)
      val expanded_mask = mask(::, None, None, ::).expand(bsz, 1, _tgt_len, src_len).to(dtype)
      val inverted_mask = 1.0 - expanded_mask
      inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

  /** LlamaRMSNorm is equivalent to T5LayerNorm */
  class LlamaRMSNorm[D <: FloatNN: Default](hidden_size: Int, eps: Float=1e-6) extends nn.Module {
    val weight = register(torch.ones(hidden_size))
    val variance_epsilon = eps

    def apply(hidden_states: Tensor[D]): Tensor[Promoted[Float32, D]] =
      val input_dtype = hidden_states.dtype
      var hidden_states_fp32 = hidden_states.to(torch.float32)
      val variance = hidden_states_fp32.pow(2).mean(-1, keepdim=true)
      hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + variance_epsilon)
      weight * hidden_states_fp32.to(input_dtype)
  }

  // case class ModelArgs(
  //     dim: Int = 512,
  //     nLayers: Int = 8,
  //     nHeads: Int = 8,
  //     vocabSize: Int = -1,
  //     multipleOf: Int = 256,
  //     normEps: Float = 1e-5,
  //     maxBatchSize: Int = 32,
  //     maxseq_len: Int = 2048
  // )



  class RMSNorm[D <: DType](val dim: Int, val eps: Float = 1e-6) extends nn.Module {
    val weight = register(torch.ones(dim))

    def apply(hidden_states: Tensor[?]) =
      val inputDtype = hidden_states.dtype
      val variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim = true)
      val updated_hidden_states = hidden_states * torch.rsqrt(variance + eps)
      (weight * updated_hidden_states).to(inputDtype)
  }

  class LlamaRotaryEmbedding(
      dim: Int,
      max_position_embeddings: Int = 2048,
      base: Int = 10000,
      device: Option[Device] = None
  ) extends nn.Module {
    var max_seq_len_cached = 0
    var cos_cached: Tensor[?] = null
    var sin_cached: Tensor[?] = null

    val invFreq = 1f / (base ** (torch.arange(0, dim, 2).float.to(device) / dim))
    registerBuffer("inv_freq", invFreq)

    protected def set_cos_sin_cache(seq_len: Int, device: Device, dtype: DType) = {
      this.max_seq_len_cached = seq_len
      val t = torch.arange(end = this.max_seq_len_cached, device = device, dtype = invFreq.dtype)

      val freqs = torch.einsum("i,j->ij", t, invFreq)
      // Different from paper, but it uses a different permutation in order to obtain the same calculation
      val emb = torch.cat(Seq(freqs, freqs), dim = -1)
      cos_cached = registerBuffer("cos_cached", emb.cos.index(None, None, ::, ::).to(dtype))
      sin_cached = registerBuffer("sin_cached", emb.sin.index(None, None, ::, ::).to(dtype))
    }

    def apply[D <: FloatNN](x: Tensor[D], seq_len: Option[Int] = None): (Tensor[D], Tensor[D]) = {
      // seq_len.filter(_ > max_seq_len_cached).foreach(seq_len => set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype))
      for len <- seq_len if len > max_seq_len_cached
      do set_cos_sin_cache(len, device = x.device, dtype = x.dtype)
      (
        this.cos_cached(::, ::, Slice(end = seq_len), ---).to(x.dtype),
        this.sin_cached(::, ::, Slice(end = seq_len), ---).to(x.dtype)
      )
    }
  }

  /** LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev */
  class LlamaLinearScalingRotaryEmbedding(
      dim: Int,
      max_position_embeddings: Int = 2048,
      base: Int = 10000,
      device: Option[Device] = None,
      scaling_factor: Float = 1.0
  ) extends LlamaRotaryEmbedding(dim, max_position_embeddings, base, device) {

    override protected def set_cos_sin_cache(seq_len: Int, device: Device, dtype: DType) = {
      max_seq_len_cached = seq_len
      val t = torch.arange(
        end = this.max_seq_len_cached,
        device = device,
        dtype = invFreq.dtype
      ) / scaling_factor

      val freqs = torch.einsum("i,j->ij", t, invFreq)
      // Different from paper, but it uses a different permutation in order to obtain the same calculation
      val emb = torch.cat(Seq(freqs, freqs), dim = -1)
      cos_cached = registerBuffer("cos_cached", emb.cos.index(None, None, ::, ::).to(dtype))
      sin_cached = registerBuffer("sin_cached", emb.sin.index(None, None, ::, ::).to(dtype))
    }
  }

  /** LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97
    * and /u/emozilla
    */
  class LlamaDynamicNTKScalingRotaryEmbedding(
      dim: Int,
      max_position_embeddings: Int = 2048,
      base: Int = 10000,
      device: Option[Device] = None,
      scaling_factor: Float = 1.0
  ) extends LlamaRotaryEmbedding(dim, max_position_embeddings, base, device) {

    override protected def set_cos_sin_cache(seq_len: Int, device: Device, dtype: DType): Unit = {
      max_seq_len_cached = seq_len

      if seq_len > this.max_position_embeddings then
        val base = this.base * (
          (scaling_factor * seq_len / this.max_position_embeddings) - (this.scaling_factor - 1)
        ) ** (dim / (this.dim - 2))
        val invFreq = 1.0 / (base ** (torch.arange(0, this.dim, 2).float().to(device) / dim))
        registerBuffer("inv_freq", invFreq)

      val t = torch.arange(end = seq_len, device = device, dtype = invFreq.dtype)

      val freqs = torch.einsum("i,j->ij", t, invFreq)
      // Different from paper, but it uses a different permutation in order to obtain the same calculation
      val emb = torch.cat(Seq(freqs, freqs), dim = -1)
      cos_cached = registerBuffer("cos_cached", emb.cos.index(None, None, ::, ::).to(dtype))
      sin_cached = registerBuffer("sin_cached", emb.sin.index(None, None, ::, ::).to(dtype))
    }
  }

  /** Rotates half the hidden dims of the input */
  def rotate_half[D <: DType](x: Tensor[D]) =
    val x1 = x(---, Slice(None, x.shape(-1) / 2))
    val x2 = x(---, Slice(x.shape(-1) / 2, None))
    torch.cat(Seq(-x2, x1), dim = -1)

  def apply_rotary_pos_emb[D <: FloatNN](
      q: Tensor[D],
      k: Tensor[D],
      cos: Tensor[D],
      sin: Tensor[D],
      position_ids: Option[
        Tensor[Int64]
      ] // TODO check whether position_ids are really an Option here
  ) =
    // The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    var _cos = cos.squeeze(1).squeeze(0) // [seq_len, dim]
    var _sin = sin.squeeze(1).squeeze(0) // [seq_len, dim]
    _cos = cos(position_ids.getOrElse(None))
      .unsqueeze(1) // [bs, 1, seq_len, dim]  // TODO should we add index/apply with Option?
    _sin = sin(position_ids.getOrElse(None)).unsqueeze(1) // [bs, 1, seq_len, dim]
    val q_embed = (q * cos) + (rotate_half(q) * sin)
    val k_embed = (k * cos) + (rotate_half(k) * sin)
    (q_embed, k_embed)

  class LlamaMLP[D <: FloatNN: Default](
      config: LlamaConfig
  ) extends nn.Module {
    val hidden_size = config.hidden_size
    val intermediate_size = config.intermediate_size
    var gate_proj = nn.Linear(hidden_size, intermediate_size, hasBias = false)
    var up_proj = nn.Linear(hidden_size, intermediate_size, hasBias = false)
    var down_proj = nn.Linear(intermediate_size, hidden_size, hasBias = false)
    val act_fn = F.silu[D] // ACT2FN(config.hidden_act) // TODO

    def apply(x: Tensor[D]): Tensor[D] =
      if config.pretraining_tp > 1 then
        val slice = intermediate_size / config.pretraining_tp
        val gate_proj_slices = this.gate_proj.weight.split(slice, dim = 0)
        val up_proj_slices = this.up_proj.weight.split(slice, dim = 0)
        val down_proj_slices = this.down_proj.weight.split(slice, dim = 1)

        val gate_proj = torch.cat(
          for i <- 0 until config.pretraining_tp yield F.linear(x, gate_proj_slices(i)),
          dim = -1
        )
        val up_proj = torch.cat(
          for i <- 0 until config.pretraining_tp yield F.linear(x, up_proj_slices(i)),
          dim = -1
        )

        val intermediate_states = (act_fn(gate_proj) * up_proj).split(slice, dim = 2)
        (for i <- 0 until config.pretraining_tp
        yield F.linear(intermediate_states(i), down_proj_slices(i)))
          .reduce(_ + _) // TODO can we implement sum for seqs of tensors via Numeric?
      else this.down_proj(act_fn(gate_proj(x)) * up_proj(x))
  }

  /** This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    *
    * The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    * num_attention_heads, seqlen, head_dim)
    */
  def repeat_kv[D <: FloatNN](hidden_states: Tensor[D], n_rep: Int): Tensor[D] = {
    val Seq(batch, num_key_value_heads, slen, head_dim) = hidden_states.shape
    if n_rep == 1 then hidden_states
    else
      val updated_hidden_states = hidden_states(::, ::, None, ::, ::).expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim
      )
      updated_hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
  }

  /** Multi-headed attention from 'Attention Is All You Need' paper */
  class LlamaAttention[D <: FloatNN: Default](config: LlamaConfig) extends nn.Module {

    val hidden_size = config.hidden_size
    val num_heads = config.num_attention_heads
    val head_dim = hidden_size / num_heads
    val num_key_value_heads = config.num_key_value_heads
    val num_key_value_groups = num_heads / num_key_value_heads
    val pretraining_tp = config.pretraining_tp
    val max_position_embeddings = config.max_position_embeddings

    if (head_dim * num_heads) != hidden_size then
      throw new IllegalArgumentException(
        s"hidden_size must be divisible by num_heads (got `hidden_size`: $hidden_size" +
          s" and `num_heads`: $num_heads)."
      )
    val q_proj = nn.Linear(hidden_size, num_heads * head_dim, hasBias = false)
    val k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, hasBias = false)
    val v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, hasBias = false)
    val o_proj = nn.Linear(num_heads * head_dim, hidden_size, hasBias = false)

    // _init_rope
    val rotary_emb = {
      config.rope_scaling match
        case None =>
          LlamaRotaryEmbedding(head_dim, max_position_embeddings = this.max_position_embeddings)
        case Some(rope_scaling) =>
          val scaling_type = rope_scaling.tpe
          val scaling_factor = rope_scaling.factor
          scaling_type match
            case Type.Linear =>
              LlamaLinearScalingRotaryEmbedding(
                head_dim,
                max_position_embeddings = max_position_embeddings,
                scaling_factor = scaling_factor
              )
            case RopeScaling.Type.Dynamic =>
              LlamaDynamicNTKScalingRotaryEmbedding(
                head_dim,
                max_position_embeddings = max_position_embeddings,
                scaling_factor = scaling_factor
              )
    }

    private def _shape[D <: DType](tensor: Tensor[D], seq_len: Int, bsz: Int) =
      tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def apply(
        hidden_states: Tensor[D],
        attention_mask: Option[Tensor[D]] = None,
        position_ids: Option[Tensor[Int64]] = None,
        past_key_value: Option[Tensor[D]] = None,
        output_attentions: Boolean = false,
        use_cache: Boolean = false
    ): (Tensor[?], Option[Tensor[D]], Option[(Tensor[D], Tensor[D])]) = {
      val Seq(bsz, q_len, _) = hidden_states.size

      var (query_states, key_states, value_states) =
        if pretraining_tp > 1 then
          val key_value_slicing = (this.num_key_value_heads * this.head_dim) / this.pretraining_tp
          val query_slices = this.q_proj.weight
            .split((this.num_heads * this.head_dim) / this.pretraining_tp, dim = 0)
          val key_slices = this.k_proj.weight.split(key_value_slicing, dim = 0)
          val value_slices = this.v_proj.weight.split(key_value_slicing, dim = 0)

          val query_states =
            val query_states =
              for i <- 0 until this.pretraining_tp yield F.linear(hidden_states, query_slices(i))
            torch.cat(query_states, dim = -1)

          val key_states =
            val key_states =
              for i <- 0 until this.pretraining_tp yield F.linear(hidden_states, key_slices(i))
            torch.cat(key_states, dim = -1)

          val value_states =
            val value_states =
              for i <- 0 until this.pretraining_tp yield F.linear(hidden_states, value_slices(i))
            torch.cat(value_states, dim = -1)

          (query_states, key_states, value_states)
        else
          val query_states = this.q_proj(hidden_states)
          val key_states = this.k_proj(hidden_states)
          val value_states = this.v_proj(hidden_states)
          (query_states, key_states, value_states)

      query_states = query_states.view(bsz, q_len, this.num_heads, this.head_dim).transpose(1, 2)
      key_states =
        key_states.view(bsz, q_len, this.num_key_value_heads, this.head_dim).transpose(1, 2)
      value_states =
        value_states.view(bsz, q_len, this.num_key_value_heads, this.head_dim).transpose(1, 2)

      var kv_seq_len = key_states.shape(-2)
      for past_key_value <- past_key_value do kv_seq_len += past_key_value(0).shape(-2)
      val (cos, sin) = this.rotary_emb(value_states, seq_len = Some(kv_seq_len))

      // TODO is there a nicer way to do this?
      val rotary_pos_emb = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
      query_states = rotary_pos_emb._1
      key_states = rotary_pos_emb._2

      for past_key_value <- past_key_value
      do
        // reuse k, v, self_attention
        key_states = torch.cat(Seq(past_key_value(0), key_states), dim = 2)
        value_states = torch.cat(Seq(past_key_value(1), value_states), dim = 2)

      val new_key_value = if use_cache then Some(key_states, value_states) else None

      // repeat k/v heads if n_kv_heads < n_heads
      key_states = repeat_kv(key_states, this.num_key_value_groups)
      value_states = repeat_kv(value_states, this.num_key_value_groups)

      var attn_weights =
        torch.matmul(query_states, key_states.transpose(2, 3)) / Tensor(head_dim).sqrt

      if attn_weights.size != Seq(bsz, this.num_heads, q_len, kv_seq_len) then
        throw new IllegalArgumentException(
          s"Attention weights should be of size ${(bsz, this.num_heads, q_len, kv_seq_len)}, but is" +
            s" ${attn_weights.size}"
        )

      for attention_mask <- attention_mask do
        if attention_mask.size != Seq(bsz, 1, q_len, kv_seq_len) then
          throw new IllegalArgumentException(
            s"Attention mask should be of size ${(bsz, 1, q_len, kv_seq_len)}, but is ${attention_mask.size}"
          )
        attn_weights = attn_weights + attention_mask

      // upcast attention to fp32
      attn_weights =
        nn.functional.softmax(attn_weights, dim = -1)(dtype = torch.float32).to(query_states.dtype)
      var attn_output = torch.matmul(attn_weights, value_states)

      if attn_output.size != Seq(bsz, this.num_heads, q_len, this.head_dim) then
        throw new IllegalArgumentException(
          s"`attn_output` should be of size ${(bsz, this.num_heads, q_len, this.head_dim)}, but is" +
            s" ${attn_output.size}"
        )

      attn_output = attn_output.transpose(1, 2).contiguous()
      attn_output = attn_output.reshape(bsz, q_len, this.hidden_size)

      if this.pretraining_tp > 1 then
        val attn_outputs = attn_output.split(this.hidden_size / this.pretraining_tp, dim = 2)
        val o_proj_slices =
          this.o_proj.weight.split(this.hidden_size / this.pretraining_tp, dim = 1)
        // TODO can we implement sum for seqs of tensors via Numeric?
        attn_output = (for i <- 0 until this.pretraining_tp
        yield F.linear(attn_outputs(i), o_proj_slices(i))).reduce(_ + _)
      else attn_output = this.o_proj(attn_output)

      (attn_output, Option.when(output_attentions)(attn_weights), new_key_value)
    }
  }

  class LlamaDecoderLayer[D <: FloatNN: Default](config: LlamaConfig) extends nn.Module {
    val hidden_size = config.hidden_size
    val self_attn = LlamaAttention(config=config)
    val mlp = LlamaMLP(config)
    val input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    val post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    /**
      *
      * @param hidden_states input to the layer of shape `(batch, seq_len, embed_dim)`
      * @param attention_mask  attention mask of size
                  `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
      * @param position_ids 
      * @param past_key_value cached past key and value projection states
      * @param output_attentions Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                  returned tensors for more detail.
      * @param use_cache If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                  (see `past_key_values`).
      */
      def apply(
          hidden_states: Tensor[D],
          attention_mask: Option[Tensor[D]] = None,
          position_ids: Option[Tensor[Int64]] = None,
          past_key_value: Option[Tensor[D]] = None,
          output_attentions: Boolean = false,
          use_cache: Boolean = false,
      ): (Tensor[Float32], Option[(Tensor[Float32], Tensor[Float32])]) =

        var residual = hidden_states

        val _hidden_states = input_layernorm(hidden_states)

        // Self Attention
        var (new_hidden_states, self_attn_weights, present_key_value) = self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        new_hidden_states = residual + hidden_states

        // Fully Connected
        residual = hidden_states
        new_hidden_states = this.post_attention_layernorm(hidden_states)
        new_hidden_states = this.mlp(hidden_states)
        new_hidden_states = residual + hidden_states

        val outputs = (hidden_states,)

        if output_attentions then
            outputs += (self_attn_weights,)

        if use_cache then
            outputs += (present_key_value,)

        outputs
    }

  /**  The bare LLaMA Model outputting raw hidden-states without any specific head on top.
   * 
   *   Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
   */
  class LlamaModel[ParamType <: FloatNN | ComplexNN: Default](config: LlamaConfig) extends nn.Module /* extends LlamaPreTrainedModel */ {
    val padding_idx = config.pad_token_id
    val vocab_size = config.vocab_size

    var embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, Some(padding_idx))
    val layers = nn.ModuleList(for _ <- 0 until config.num_hidden_layers yield LlamaDecoderLayer(config))
    val norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    // this.gradient_checkpointing = false TODO
    // Initialize weights and apply final processing
    // this.post_init() TODO

    def get_input_embeddings = embed_tokens

    def set_input_embeddings(value: nn.Embedding[ParamType]) = embed_tokens = value

    // Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(attention_mask: Tensor[Int64], input_shape, inputs_embeds, past_key_values_length) =
        // create causal mask
        // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        // var combined_attention_mask = None
        val combined_attention_mask =
          if input_shape(-1) > 1 then
            val combined_attention_mask = make_causal_mask(
              input_shape,
              inputs_embeds.dtype,
              device=inputs_embeds.device,
              past_key_values_length=past_key_values_length,
            )

        for attention_mask <- attention_mask do
          //  [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
          val expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
              inputs_embeds.device
          )
          val combined_attention_mask = (
            if combined_attention_mask is None then expanded_attn_mask else expanded_attn_mask + combined_attention_mask
          )

        combined_attention_mask

    def forward(
        input_ids: Option[Tensor[Int64]], // = None,
        attention_mask: Option[Tensor[Float32]] = None,
        position_ids: Option[Tensor[Int64]] = None,
        past_key_values: Option[List[Tensor[Float32]]] = None,
        inputs_embeds: Option[Tensor[Float32]] = None,
        use_cache: Option[Boolean] = None,
        output_attentions: Option[Boolean] = None,
        output_hidden_states: Option[Boolean] = None,
    ): BaseModelOutputWithPast =
        val _output_attentions = output_attentions.getOrElse(this.config.output_attentions)
        val _output_hidden_states = (
            output_hidden_states.getOrElse(this.config.output_hidden_states)
        )
        val _use_cache = use_cache.getOrElse(this.config.use_cache)

      // retrieve input_ids and inputs_embeds

        //val Seq(batch_size, seq_length) 
        val x = (input_ids, inputs_embeds) match {
          case (Some(_), Some(_)) => throw new IllegalArgumentException("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
          case (Some(input_ids), _) => input_ids.shape
          case (_, Some(inputs_embeds)) => inputs_embeds.shape
          case _ => throw new IllegalArgumentException("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        }


        var seq_length_with_past = seq_length
        var past_key_values_length = 0

        past_key_values.foreach {past_key_values =>
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        }

        if position_ids is None then
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None then
            inputs_embeds = this.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None then
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = this._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if this.gradient_checkpointing and this.training then
            if use_cache then
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=false`..."
                )
                use_cache = false

        // decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(this.layers)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if this.gradient_checkpointing && this.training then

                def create_custom_forward(module) =
                    def custom_forward(*inputs) =
                        // None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache then
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions then
                all_self_attns += (layer_outputs[1],)

        hidden_states = this.norm(hidden_states)

        // add hidden states from the last decoder layer
        if output_hidden_states then
            all_hidden_states += (hidden_states,)

        next_cache = if use_cache then next_decoder_cache else None
        BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
}
}
