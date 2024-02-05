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

package gpt

/*
import torch
import torch.nn as nn
from torch.nn import functional as F
 */

// cSpell: ignore gpt, hyperparameters, logits, softmax
// cSpell: ignore CUDA, torchvision
// cSpell: ignore dtype
// cSpell: ignore stoi, itos
// cSpell: ignore nn, probs, numel, itemsize, nbytes
// cSpell: ignore xbow, xprev, isinstance, idx, tok_emb
// cSpell: ignore Andrej, Karpathy

import java.nio.file.Files
import java.net.URI

import scala.util.Using
import scala.collection.immutable.SortedSet

import org.bytedeco.javacpp.PointerScope

import torch.*
import torch.Device.CUDA
import torch.Device.CPU
import torch.nn.functional as F
import torch.nn.modules.Module
import torch.Slice
import org.bytedeco.javacpp.Pointer

import gpt.Utils.Modules as UtilsM
import gpt.Utils.CUDAMemory as Mem
import scala.annotation.nowarn

/** This is code translated from Andrej Karpathy's
  * [[video https://www.youtube.com/watch?v=kCc8FmEb1nY]] titled "Let's build GPT: from scratch, in
  * code, spelled out." The video also provides links to:
  *
  *   - [[A Google colab for the video https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing]]
  *   - [[GitHub repo for the video https://github.com/karpathy/ng-video-lecture]]
  *
  * This code tries to replicate the
  * [[original PyTorch code https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py]]
  *
  * Unfortunately the results are not on par with the original. The code works best when weight
  * initialization is **not** used. The number of epochs learned must be significantly larger (2
  * orders of magnitude) above the original. As per the `gpt.py` code, weight initialization
  * produces much worse results.
  *
  * @see
  *   https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
  */
object V2:

  /*
  # hyperparameters
  batch_size = 64 # how many independent sequences will we process in parallel?
  block_size = 256 # what is the maximum context length for predictions?
  max_iters = 5000
  eval_interval = 500
  learning_rate = 3e-4
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  eval_iters = 200
  n_embd = 384
  n_head = 6
  n_layer = 6
  dropout = 0.2
  # ------------
   */

  val batch_size = 64 // how many independent sequences will we process in parallel?
  val block_size = 256 // what is the maximum context length for predictions?
  val max_iters = 5000
  val eval_interval = 500
  val learning_rate = 3e-4
  val device = if torch.cuda.isAvailable then CUDA else CPU
  val eval_iters = 200
  val n_embed = 384
  val n_head = 6
  val n_layer = 6
  val dropout = 0.2

  // Initialize weights
  val init_mean = 0.0 // 0,0
  val init_std = 0.02 // 0.02 (original) // 0.01, 0.03

  /*
  torch.manual_seed(1337)
   */
  torch.manualSeed(1337)

  /*
  # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  with open('input.txt', 'r', encoding='utf-8') as f:
      text = f.read()
   */
  val DATA = "data"
  val INPUT = "input.txt"
  val URL_ =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

  val dataDir = os.pwd / DATA
  if !os.exists(dataDir)
  then
    println(s"Creating folder: $dataDir")
    os.makeDir(dataDir)

  val dataFile = dataDir / "input.txt"
  if !os.exists(dataFile)
  then
    // Assume the URL is valid
    val uri = URI.create(URL_)
    val url = uri.toURL() // URL(URL_)
    println(s"Downloading from $url to $dataFile")
    val nuBytes = Using.resource(url.openStream()) { inputStream =>
      // os.write(dataFile, inputStream)
      Files.copy(inputStream, dataFile.toNIO)
    }
    println(s"Read ${nuBytes} bytes")
  else println(s"File $dataFile already exists.")

  val text = os.read(dataFile)

  /*
  # here are all the unique characters that occur in this text
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  # create a mapping from characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
  decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
   */

  // here are all the unique characters that occur in this text
  val chars = SortedSet(text: _*)
  println(s"chars = ${chars.mkString(", ")}")
  val vocab_size = chars.size
  println(s"vocab_size = $vocab_size")

  // create a mapping from characters to integers
  val stoi = chars.zipWithIndex.map((ch, i) => ch -> i.toLong).toMap
  val itos = stoi.map((ch, i) => i -> ch)
  def encode(s: String) = s.map(c => stoi(c))
  def decode(s: Seq[Long]) = s.map(i => itos(i)).mkString

  println(s""""BiGram!" = "${decode(encode("BiGram!"))}"""")

  /*
  # Train and test splits
  data = torch.tensor(encode(text), dtype=torch.long)
  n = int(0.9*len(data)) # first 90% will be train, rest val
  train_data = data[:n]
  val_data = data[n:]
   */

  // Train and test splits
  val data = torch.Tensor(encode(text)).long
  val n = (0.9 * Utils.len(data)).toInt // first 90% will be train, rest val
  val train_data = data(Slice(None, n))
  val val_data = data(Slice(n, None))

  /*
  # data loading
  def get_batch(split):
      # generate a small batch of data of inputs x and targets y
      data = train_data if split == 'train' else val_data
      ix = torch.randint(len(data) - block_size, (batch_size,))
      x = torch.stack([data[i:i+block_size] for i in ix])
      y = torch.stack([data[i+1:i+block_size+1] for i in ix])
      x, y = x.to(device), y.to(device)
      return x, y
   */

  // data loading
  def getBatch(split: String) =
    // generate a small batch of data of inputs x and targets y
    val data = if split == "train" then train_data else val_data
    val ix = torch.randint(0, Utils.len(data) - block_size, Seq(batch_size)).to(dtype = int32)
    val stacks_x = ix.toSeq.map(i => data(Slice(i, i + block_size)))
    val x = torch.stack(stacks_x)
    val stacks_y = ix.toSeq.map(i => data(Slice(i + 1, i + block_size + 1)))
    val y = torch.stack(stacks_y)
    (x.to(device), y.to(device))

  val (xb, yb) = getBatch("train")
  println("xb:")
  println(decode(xb(0).toSeq))
  println("yb:")
  println(decode(yb(0).toSeq))

  trait BigramLanguageModel extends nn.Module:
    def forward(
        idx: Tensor[Int64],
        targets: Option[Tensor[Int64]] = None
    ): (Tensor[Float32], Tensor[Float32])
    def generate(idx: Tensor[Int64], max_new_tokens: Int): Tensor[Int64]
    def apply(x: Tensor[Int64], y: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])
    def apply(x: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])

  /*
  @torch.no_grad()
  def estimate_loss():
      out = {}
      model.eval()
      for split in ['train', 'val']:
          losses = torch.zeros(eval_iters)
          for k in range(eval_iters):
              X, Y = get_batch(split)
              logits, loss = model(X, Y)
              losses[k] = loss.item()
          out[split] = losses.mean()
      model.train()
      return out
   */

  def estimateLoss(model: BigramLanguageModel) =
    val out = scala.collection.mutable.Map[String, Float]()
    model.eval()

    for split <- List("train", "val")
    do
      // println(s"Estimate '$split' loss")
      val losses: Tensor[Float32] = torch.zeros(eval_iters, device = device)
      for k <- 0 until eval_iters
      do
        // Need to release memory
        Using.resource(new PointerScope()) { p =>
          val (x, y) = getBatch(split)
          val (logits, loss) = model(x, y)
          losses(Seq(k)) = loss.item
        }

      out(split) = losses.mean.item

    model.train()
    out

  /*
  class Head(nn.Module):
      """ one head of self-attention """

      def __init__(self, head_size):
          super().__init__()
          self.key = nn.Linear(n_embd, head_size, bias=False)
          self.query = nn.Linear(n_embd, head_size, bias=False)
          self.value = nn.Linear(n_embd, head_size, bias=False)
          self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          # input of size (batch, time-step, channels)
          # output of size (batch, time-step, head size)
          B,T,C = x.shape
          k = self.key(x)   # (B,T,hs)
          q = self.query(x) # (B,T,hs)
          # compute attention scores ("affinities")
          wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
          wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
          wei = F.softmax(wei, dim=-1) # (B, T, T)
          wei = self.dropout(wei)
          # perform the weighted aggregation of the values
          v = self.value(x) # (B,T,hs)
          out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
          return out
   */

  class Head[D <: FloatNN: Default](
      n_embed: Int,
      head_size: Int,
      block_size: Int,
      dropout: Double
  ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    val key = register(nn.Linear[D](n_embed, head_size, addBias = false))
    val query = register(nn.Linear[D](n_embed, head_size, addBias = false))
    val value = register(nn.Linear[D](n_embed, head_size, addBias = false))
    val ones = torch.ones[D](Seq(block_size, block_size), dtype = key.paramType)
    val tril = registerBuffer(torch.tril(ones), "tril")
    val drop = register(nn.Dropout(dropout))

    def forward(x: Tensor[D]): Tensor[D] =
      // input of size (batch, time-step, channels) = (B,T,C)
      // output of size (batch, time-step, head size) = (B,T,H)
      // batch, number of time steps, channels
      val Seq(b, t, c) = x.shape
      assert(block_size == t, "Block size must be equal to time step")
      val k = key(x) // (B,T,C) @ (C,H) -> (B,T,H)
      val q = query(x) // (B,T,H)
      // compute attention scores ("affinities")
      // hs should be the head size
      val hs = k.size.last
      assert(head_size == hs, "Head size does not match k")
      val qk =
        q `@` k.transpose(-2, -1) * Tensor(hs)
          .pow(-0.5)
          .to(dtype = q.dtype) // (B, T, H) @ (B, H, T) -> (B, T, T)
      val mask = qk
        .maskedFill(tril(Slice(None, n), Slice(None, n)) == 0, Float.NegativeInfinity)
        .to(dtype = q.dtype) // (B, T, T)
      val soft = F.softmax(mask, dim = -1) // (B, T, T)
      val wei = drop(soft)
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,H)
      val out = wei `@` v // (B, T, T) @ (B, T, H) -> (B, T, H)
      out

    def apply(x: Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String =
      s"${getClass.getSimpleName()}(n_embed=$n_embed, head_size=$head_size, block_size=$block_size)"
  end Head

  class Head_2[D <: FloatNN: Default](
      nEmbed: Int,
      headSize: Int,
      blockSize: Int
      // drop: Double
  ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    val key = register(nn.Linear[D](nEmbed, headSize, addBias = false))
    val query = register(nn.Linear[D](nEmbed, headSize, addBias = false))
    val value = register(nn.Linear[D](nEmbed, headSize, addBias = false))
    val ones = torch.ones[D](Seq(blockSize, blockSize), dtype = key.paramType)
    val tril = registerBuffer(torch.tril(ones), "tril")
    val drop = register(nn.Dropout(dropout))

    def forward(x: Tensor[D]): Tensor[D] =
      // input of size (batch, time-step, channels) = (B,T,C)
      // output of size (batch, time-step, head size) = (B,T,H)
      // batch, number of time steps, channels
      val Seq(b, t, c) = x.shape
      // fails on generate ?
      // assert(blockSize == t, "Block size must be equal to time step")

      // key = Linear(inFeatures=C, outFeatures=T, bias=false)
      val k = key(x) // (B,T,C) @ (C,H) -> (B,T,H)
      val q = query(x) // (B,T,H)
      // compute attention scores ("affinities")
      // c should be the head size
      val hs = k.size.last
      assert(headSize == hs, "Head size does not match k")
      val qk =
        q `@` k.transpose(-2, -1) * Tensor(hs)
          .pow(-0.5)
          .to(dtype = q.dtype) // (B, T, H) @ (B, H, T) -> (B, T, T)
      // val mask = qk.maskedFill(tril == 0, Float.NegativeInfinity) // (B, T, T)
      // val mask = qk.maskedFill(tril((º`:`n), (º`:`n)) == 0, Float.NegativeInfinity).to(dtype=q.dtype) // (B, T, T)
      val mask = qk
        .maskedFill(tril(Slice(None, n), Slice(None, n)) == 0, Float.NegativeInfinity)
        .to(dtype = q.dtype) // (B, T, T)
      // val softmax = F.softmax(mask, dim= -1) // (B, T, T)
      // val wei = dropout(softmax)
      val soft = F.softmax(mask, dim = -1) // (B, T, T)
      val wei = drop(soft)
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,H)
      val out = wei `@` v // (B, T, T) @ (B, T, H) -> (B, T, H)
      out

    def apply(x: Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String =
      s"${getClass.getSimpleName()}(n_embed=$nEmbed, head_size=$headSize, block_size=$blockSize)"
  end Head_2

  /*
  class MultiHeadAttention(nn.Module):
      """ multiple heads of self-attention in parallel """

      def __init__(self, num_heads, head_size):
          super().__init__()
          self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
          self.proj = nn.Linear(head_size * num_heads, n_embd)
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          out = torch.cat([h(x) for h in self.heads], dim=-1)
          out = self.dropout(self.proj(out))
          return out
   */

  class MultiHeadAttention[D <: FloatNN: Default](
      numHeads: Int,
      nEmbed: Int,
      headSize: Int,
      blockSize: Int,
      dropout: Double
  ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    // Cannot register with the same name
    // val hs = 0 until numHeads map{ _ => register(Head_2(nEmbed, headSize, blockSize)) }
    val hs = 0 until numHeads map { i =>
      Utils.register_i(this, Head_2(nEmbed, headSize, blockSize), i)
    }
    // val hs = 0 until numHeads map{ i => Utils.register_i(this, Head(nEmbed, headSize, blockSize, dropout), i) }
    val heads = register(nn.ModuleList(hs: _*))
    // TODO: BUG - self.proj = nn.Linear(head_size * num_heads, n_embd)
    val proj = register(nn.Linear(headSize * numHeads, nEmbed))
    // val proj = register( nn.Linear(nEmbed, nEmbed) )
    val drop = register(nn.Dropout(dropout))

    def forward(x: Tensor[D]): Tensor[D] =
      // torch.cat(heads.modules.map( h => h(x) ), dim=1)
      val out = torch.cat(heads.map(h => h(x)).toSeq, dim = -1)
      drop(proj(out))

    def apply(x: Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String =
      s"${getClass().getSimpleName()}(numHeads=$numHeads, nEmbed=$nEmbed, headSize=$headSize, blockSize=$blockSize)"

  end MultiHeadAttention

  /*
  class FeedFoward(nn.Module):
      """ a simple linear layer followed by a non-linearity """

      def __init__(self, n_embd):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(n_embd, 4 * n_embd),
              nn.ReLU(),
              nn.Linear(4 * n_embd, n_embd),
              nn.Dropout(dropout),
          )

      def forward(self, x):
          return self.net(x)
   */
  class FeedForward[D <: FloatNN: Default](
      nEmbed: Int
  ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:
    val net = register(
      nn.Sequential(
        // Increase output dimension by 4
        nn.Linear(nEmbed, 4L * nEmbed),
        nn.ReLU(),
        // Decrease output dimension by 4
        nn.Linear(
          4L * nEmbed,
          nEmbed
        ), // residual network implemented using projection - why not on x directly?
        nn.Dropout(dropout)
      )
    )

    def forward(x: Tensor[D]): Tensor[D] = net(x)

    def apply(x: Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(nEmbed = $nEmbed)"

  end FeedForward

  /*
  class Block(nn.Module):
      """ Transformer block: communication followed by computation """

      def __init__(self, n_embd, n_head):
          # n_embd: embedding dimension, n_head: the number of heads we'd like
          super().__init__()
          head_size = n_embd // n_head
          self.sa = MultiHeadAttention(n_head, head_size)
          self.ffwd = FeedFoward(n_embd)
          self.ln1 = nn.LayerNorm(n_embd)
          self.ln2 = nn.LayerNorm(n_embd)

      def forward(self, x):
          x = x + self.sa(self.ln1(x))
          x = x + self.ffwd(self.ln2(x))
          return x
   */
  class Block[D <: FloatNN: Default](
      nEmbed: Int,
      nHead: Int,
      blockSize: Int,
      // TODO check if we need this
      @nowarn("msg=unused explicit parameter")
      vocabSize: Int,
      dropout: Double
  ) extends torch.nn.modules.TensorModule[D]:

    // n_embd: embedding dimension, n_head: the number of heads we'd like
    val headSize = nEmbed / nHead
    val sa = register(
      MultiHeadAttention(
        numHeads = nHead,
        nEmbed = nEmbed,
        headSize = headSize,
        blockSize = blockSize,
        dropout = dropout
      )
    )
    val ffwd = register(FeedForward(nEmbed))
    // val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    val ln1 = register(nn.LayerNorm(Seq(nEmbed)))
    val ln2 = register(nn.LayerNorm(Seq(nEmbed)))

    def forward(x: Tensor[D]): Tensor[D] =
      val x1 = x + sa(ln1(x))
      // TODO: BUG found
      x1 + ffwd(ln2(x1))

    def apply(x: Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(nEmbed = $nEmbed)"

  end Block

  /*
  class GPTLanguageModel(nn.Module):

      def __init__(self):
          super().__init__()
          # each token directly reads off the logits for the next token from a lookup table
          self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
          self.position_embedding_table = nn.Embedding(block_size, n_embd)
          self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
          self.ln_f = nn.LayerNorm(n_embd) # final layer norm
          self.lm_head = nn.Linear(n_embd, vocab_size)

          # better init, not covered in the original GPT video, but important, will cover in followup video
          self.apply(self._init_weights)

      def _init_weights(self, module):
          if isinstance(module, nn.Linear):
              torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
              if module.bias is not None:
                  torch.nn.init.zeros_(module.bias)
          elif isinstance(module, nn.Embedding):
              torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

      def forward(self, idx, targets=None):
          B, T = idx.shape

          # idx and targets are both (B,T) tensor of integers
          tok_emb = self.token_embedding_table(idx) # (B,T,C)
          pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
          x = tok_emb + pos_emb # (B,T,C)
          x = self.blocks(x) # (B,T,C)
          x = self.ln_f(x) # (B,T,C)
          logits = self.lm_head(x) # (B,T,vocab_size)

          if targets is None:
              loss = None
          else:
              B, T, C = logits.shape
              logits = logits.view(B*T, C)
              targets = targets.view(B*T)
              loss = F.cross_entropy(logits, targets)

          return logits, loss

      def generate(self, idx, max_new_tokens):
          # idx is (B, T) array of indices in the current context
          for _ in range(max_new_tokens):
              # crop idx to the last block_size tokens
              idx_cond = idx[:, -block_size:]
              # get the predictions
              logits, loss = self(idx_cond)
              # focus only on the last time step
              logits = logits[:, -1, :] # becomes (B, C)
              # apply softmax to get probabilities
              probs = F.softmax(logits, dim=-1) # (B, C)
              # sample from the distribution
              idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
              # append sampled index to the running sequence
              idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
          return idx
   */

  // https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  class GPTLanguageModel(
      vocabSize: Int,
      blockSize: Int,
      nEmbed: Int,
      nBlocks: Int,
      nHead: Int,
      dropout: Double
  ) extends BigramLanguageModel:

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register(nn.Embedding(vocabSize, nEmbed))
    val position_embedding_table = register(nn.Embedding(blockSize, nEmbed))
    val blocks_i = 0 until nBlocks map { i => Block(nEmbed, nHead, blockSize, vocabSize, dropout) }
    val blocks = register(nn.Sequential(blocks_i: _*))
    val ln_f = register(nn.LayerNorm(Seq(nEmbed)))
    val lm_head = register(nn.Linear(nEmbed, vocabSize))

    // better init, not covered in the original GPT video, but important, will cover in followup video
    // self.apply(self._init_weights)
    // modules.foreach(init_weights)

    /*
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
     */
    @nowarn("msg=unused private member")
    private def init_weights(m: Module): Unit =
      m match
        case lm: nn.Linear[?] =>
          torch.nn.init.normal_(lm.weight, mean = init_mean, std = init_std)
          if lm.hasBias()
          then torch.nn.init.zeros_(lm.bias)
        case em: nn.Embedding[?] =>
          torch.nn.init.normal_(em.weight, mean = init_mean, std = init_std)
        case _ => ()

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b, t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table(idx) // (B,T,C) where C is nEmbed
      // positions of tokens
      val pos = torch.arange(0L, t, device = device) // (T) were T is the block size?
      val pos_embed = position_embedding_table(pos) // (T,C)
      // Add the position embeddings
      val x0 = token_embed + pos_embed // (B,T,C)
      val x1 = blocks(x0) // apply blocks of self-attention (B,T,C) - C = nEmbed
      val x2 = ln_f(x1) // (B,T,C) layer norm
      val logits = lm_head(x2) // (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

      if targets.isEmpty
      then
        val zero = torch.Tensor(0.0f)
        (logits, zero)
      else
        val Seq(b, t, c) = logits.shape
        val logitsV = logits.view(b * t, c) // batch size x number of classes
        val targetsV = targets.get.view(b * t)
        // NOTE: we are using all of the time-steps (symbols) to calculate error
        // Why do we not use only the last one, the prediction?
        val loss = F.crossEntropy(logitsV, targetsV)
        (logitsV, loss)

    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      Mem.printMemoryInfo(0)

      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        // crop idx to the last block_size tokens
        val idx_cond = idx_(Slice(), Slice(-blockSize, None))
        // get the predictions
        val (logits_t, loss) = apply(idx_cond)
        // focus only on the last time step
        val logits = logits_t(Slice(), -1, Slice()) // becomes (B, C)
        // apply softmax to get probabilities
        val probs = F.softmax(logits, dim = -1L) // (B, C)
        // sample from the distribution
        val idx_next = torch.multinomial(probs, numSamples = 1) // (B, 1)
        // append sampled index to the running sequence
        idx_ = torch.cat(Seq(idx_, idx_next), dim = 1) // (B, T+1)
        Mem.printMemoryInfo(0)
      idx_

    def apply(x: Tensor[Int64], y: Tensor[Int64]) =
      forward(x, Some(y))

    def apply(x: Tensor[Int64]) =
      forward(x, None)

  end GPTLanguageModel

  /*
  for iter in range(max_iters):

      # every once in a while evaluate the loss on train and val sets
      if iter % eval_interval == 0 or iter == max_iters - 1:
          losses = estimate_loss()
          print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      # sample a batch of data
      xb, yb = get_batch('train')

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

   */

  def train(m: BigramLanguageModel, learningRate: Double, maxIterations: Int): Unit =
    m.to(device)

    // print the number of parameters in the model
    val nuParams = m.parameters.map(_.numel).sum
    // println(s"${nuParams/1e6}M parameters")
    println(s"Device = ${device}")
    println(s"${nuParams} parameters")
    println(s"learningRate = ${learningRate}")
    println(s"maxIterations = ${maxIterations}")
    println(s"dropout = ${dropout}")
    println(s"init_mean = ${init_mean}")
    println(s"init_std = ${init_std}")
    val nBytes = m.parameters.map(t => t.native.nbytes()).sum
    println(s"${nuParams} parameters = ${nBytes} bytes = ${Utils.humanReadableSize(nBytes)}")

    m.train()
    // create a PyTorch optimizer
    val optimizer = torch.optim.AdamW(m.parameters, lr = learningRate)

    var delta = 0L
    var total = 0L
    for iter <- 0 until maxIterations
    do
      // make sure we deallocate intermediate tensors in time
      Using.resource(new PointerScope()) { p =>

        // every once in a while evaluate the loss on train and val sets
        if (iter % eval_interval == 0) || (iter == maxIterations - 1)
        then
          val losses = estimateLoss(m)
          val memoryBytes = Utils.humanReadableSize(Pointer.physicalBytes())
          delta = delta / eval_interval
          val accumulated = Utils.humanReadableDuration(total)
          val perIteration = Utils.humanReadableDuration(delta)
          println(
            s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}, mem $memoryBytes @ ${accumulated}, mean $perIteration"
          )
          // printMemoryInfo(device.index)
          // printMemoryInfo(0)
          delta = 0L

        val elapsed = Utils.elapsedOnly {
          // sample a batch of datas
          val (xb, yb) = getBatch("train")

          // evaluate the loss
          val (logits, loss) = m(xb, yb)
          optimizer.zeroGrad(setToNone = true)
          loss.backward()
          optimizer.step()
        }
        delta = delta + elapsed
        total = total + elapsed
      }

    val losses = estimateLoss(m)
    val accumulated = Utils.humanReadableDuration(total)
    val perIteration = Utils.humanReadableDuration(total / maxIterations)
    println(
      s"step ${maxIterations}: train loss ${losses("train")}, val loss ${losses("val")}, @ ${accumulated}, mean $perIteration"
    )

  // sbt "set ThisBuild / enableGPU := true" "examples/runMain gpt.V2"
  // nohup sbt "set ThisBuild / enableGPU := true" "examples/runMain gpt.V2" > tmp.txt 2>&1 &
  def main(args: Array[String]): Unit =
    println("V2")

    torch.manualSeed(1337)

    /*
    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
     */

    val model = GPTLanguageModel(
      vocabSize = vocab_size,
      blockSize = block_size,
      nEmbed = n_embed,
      nBlocks = n_layer,
      nHead = n_head,
      dropout = dropout
    )
    println(Utils.totalNuParameters(model))
    println(UtilsM.moduleInfoString(model))
    // Mem.printMemoryInfo(device.index)
    Mem.printMemoryInfo(0)

    /*
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    See train loop above
     */

    // train(model, learning_rate, 67000)
    train(model, 1e-4, 41500) // with weight init

    /*
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
     */

    // TODO: Bug just (1,1) ?
    val context = torch.zeros(Seq(1, block_size), dtype = torch.int64, device = device)
    // val context = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device)
    val embedding = model.generate(idx = context, max_new_tokens = 500)(0)
    val decoded = decode(embedding.toSeq)
    println(s"decode:'$decoded'")

    ()

end V2
