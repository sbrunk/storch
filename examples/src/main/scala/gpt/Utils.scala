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

// cSpell:ignore CUDA

import torch.Tensor
import torch.nn.modules.Module
import org.bytedeco.pytorch.cuda.Stat
import org.bytedeco.pytorch.global.torch_cuda

/** Utility functions
  */
object Utils:

  def len[T <: torch.DType](t: Tensor[T]): Int =
    // t.size.sum
    t.shape.sum

  def register_i[M1 <: Module, M2 <: Module](parent: M1, child: M2, i: Int, n: String = "")(using
      name: sourcecode.Name
  ): M2 =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    val name_i = s"${name_}_$i"
    parent.register(child, name_i)

  def totalNuParameters(m: Module): String =
    val nuParams = m.parameters.map(_.numel).sum
    if nuParams < 1e5
    then s"${nuParams} parameters"
    else if nuParams < 1e6
    then s"${nuParams / 1e3}K parameters"
    else s"${nuParams / 1e6}M parameters"

  val SI = (BigInt(1000), Vector("B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"))
  val BINARY = (BigInt(1024), Vector("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"))

  /** Converts a number of bytes into a human-readable string such as `2.2 MB` or `8.0 EiB`.
    *
    * @param bytes
    *   the number of bytes we want to convert
    * @param si
    *   if true, we use base 10 SI units where 1000 bytes are 1 kB. If false, we use base 2 IEC
    *   units where 1024 bytes are 1 KiB.
    * @return
    *   the bytes as a human-readable string
    *
    * @see
    *   https://stackoverflow.com/questions/45885151/bytes-in-human-readable-format-with-idiomatic-scala
    * @see
    *   https://stackoverflow.com/questions/3758606/how-can-i-convert-byte-size-into-a-human-readable-format-in-java
    */
  def humanReadableSize(bytes: BigInt, si: Boolean = false): String =
    // See https://en.wikipedia.org/wiki/Byte
    val (baseValue, unitStrings) =
      if (si)
        SI
      else
        BINARY

    def getExponent(curBytes: BigInt, baseValue: BigInt, curExponent: Int = 0): Int =
      if (curBytes < baseValue)
      then curExponent
      else
        val newExponent = 1 + curExponent
        // getExponent(curBytes / (baseValue * newExponent), baseValue, newExponent)
        getExponent(curBytes / baseValue, baseValue, newExponent)

    val exponent = getExponent(bytes, baseValue)
    val divisor = baseValue.pow(exponent)
    val unitString = unitStrings(exponent)

    // Divide the bytes and show one digit after the decimal point
    f"${bytes.toDouble / divisor.toDouble}%.1f $unitString"

  inline def elapsed[R](inline block: => R): (Long, R) = {
    val t0 = System.nanoTime()
    val r = block
    val t1 = System.nanoTime()
    val elapsed = t1 - t0
    // elapsed time in nanoseconds
    (elapsed, r)
  }

  inline def elapsedOnly[R](inline block: => R): Long = elapsed(block)._1

  def durationParts(nanoSeconds: Long) =
    val duration = java.time.Duration.ofNanos(nanoSeconds)
    val d = duration.toDaysPart()
    val hh = duration.toHoursPart()
    val mm = duration.toMinutesPart()
    val ss = duration.toSecondsPart()
    val ms = duration.toMillisPart()
    val ns = duration.toNanosPart()
    (d, hh, mm, ss, ms, ns)

  def humanReadableDuration(nanoSeconds: Long) =
    val (d, hh, mm, ss, ms, ns) = durationParts(nanoSeconds)
    String.format("%02d %02d:%02d:%02d.%03d", d, hh, mm, ss, ms)

  object Modules:

    def moduleName(m: Module): String =
      // m.getClass().getSimpleName()
      m.toString()

    def moduleClass(m: Module): String =
      m.getClass().getSimpleName()

    /** Collects information of a module and returns this as a string.
      *
      * Complex modules are shown hierarchically. Information includes the modules `toString` output
      * that usually holds the variable name and class parameter values. We also add the number of
      * tensor parameter amd their value in the leaf modules. For the other modules the sum of the
      * number of tensor parameters are shown.
      *
      * Use this output to "debug" your networks
      *
      * @param m
      * @return
      *   string
      */
    def doModuleInfoString(m: Module, indent: Int): String =
      val parametersCount = m.parameters.size
      if m.modules.isEmpty
      then
        val parametersSize = m.parameters.map(_.numel).mkString("<", ",", ">")
        val thisModule = s"${moduleName(m)}: #$parametersCount $parametersSize "
        thisModule
      else
        val parametersSize = m.parameters.map(_.numel).sum
        val thisModule = s"${moduleName(m)}: #$parametersCount $parametersSize "
        thisModule + m.namedChildren
          .map((name, module) =>
            s"${" " * (indent + 2)}$name: " + doModuleInfoString(module, indent + 2)
          )
          .mkString("(\n", "\n", s"\n${" " * indent})")

    /** Collects information of a module and returns this as a string.
      *
      * Complex modules are shown hierarchically. Information includes the modules `toString` output
      * that usually holds the variable name and class parameter values. We also add the number of
      * tensor parameter amd their value in the leaf modules. For the other modules the sum of the
      * number of tensor parameters are shown.
      *
      * Use this output to "debug" your networks
      *
      * @param m
      * @return
      *   string
      */
    def moduleInfoString(m: Module): String =
      doModuleInfoString(m, 0)

  end Modules

  object CUDAMemory:

    def statToDict(stat: Stat, name: String, dict: scala.collection.mutable.Map[String, Long]) =
      dict(s"$name.current") = stat.current()
      dict(s"$name.peak") = stat.peak()
      dict(s"$name.allocated") = stat.allocated()
      dict(s"$name.freed") = stat.freed()

    def statArrayToDict(
        statArray: Stat,
        name: String,
        dict: scala.collection.mutable.Map[String, Long]
    ) =

      val statTypeNames = Array("all", "small_pool", "large_pool")
      for i <- 0 until statTypeNames.length
      do statToDict(statArray.position(i), s"$name.${statTypeNames(i)}", dict)

    /** Equivalent to PyTorch memory_stats. Returns a dictionary of CUDA memory allocator statistics
      * for a given device. The return value of this function is a dictionary of statistics, each of
      * which is a non-negative integer.
      *
      * Reference implementation
      * @see
      *   https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
      * @see
      *   https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/Module.cpp#L1243
      * @see
      *   https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py#L165
      *
      * JavaCPP references
      * @see
      *   https://github.com/bytedeco/javacpp-presets/blob/master/pytorch/src/gen/java/org/bytedeco/pytorch/cuda/DeviceStats.java#L26
      * @see
      *   https://github.com/bytedeco/javacpp-presets/blob/master/pytorch/src/gen/java/org/bytedeco/pytorch/global/torch_cuda.java#L766
      * @see
      *   https://github.com/bytedeco/javacpp-presets/issues/1422
      *
      * @param device
      * @return
      */
    def memoryStats(device: Int): scala.collection.mutable.Map[String, Long] =
      // get the device
      val cudaAllocator = torch_cuda.getAllocator()
      println(cudaAllocator.initialized())
      val stats = cudaAllocator.getDeviceStats(device)

      // Collect the statistics
      val result = scala.collection.mutable.Map[String, Long]()
      result("num_alloc_retries") = stats.num_alloc_retries
      result("num_ooms") = stats.num_ooms
      result("max_split_size") = stats.max_split_size

      // Stat(stats.allocation) casts a Pointer to a Stats pointer. Can be an array
      statArrayToDict(Stat(stats.allocation), "allocation", result)
      statArrayToDict(Stat(stats.segment), "segment", result)
      statArrayToDict(Stat(stats.active), "active", result)
      statArrayToDict(Stat(stats.inactive_split), "inactive_split", result)
      statArrayToDict(Stat(stats.allocated_bytes), "allocated_bytes", result)
      statArrayToDict(Stat(stats.reserved_bytes), "reserved_bytes", result)
      statArrayToDict(Stat(stats.active_bytes), "active_bytes", result)
      statArrayToDict(Stat(stats.inactive_split_bytes), "inactive_split_bytes", result)
      statArrayToDict(Stat(stats.requested_bytes), "requested_bytes", result)

      statToDict(stats.oversize_allocations, "oversize_allocations", result)
      statToDict(stats.oversize_segments, "oversize_segments", result)
      result

    def memory_summary(device: Int, abbreviated: Boolean = false) =
      val result = memoryStats(device)
      val l = result.toList.sortBy(_._1)
      l.map((a, b) => s"$a : ${Utils.humanReadableSize(b)}")
        .mkString("\n")

    def printMemoryInfo(device: Int) =
      println(memory_summary(device))

  end CUDAMemory

end Utils
