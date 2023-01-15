---
sidebar_position: 3
---

# GPU Support

Storch supports GPU accelerated tensor operations. For Nvidia GPUs, it should mostly work out of the box.

## Enable GPU support

The easiest and most portable way to enable GPU support is via the PyTorch platform dependency:

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
  <TabItem value="sbt" label="sbt" default>

```scala
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
libraryDependencies += Seq(
  "dev.storch" % "storch:@VERSION@",
  "org.bytedeco" % "pytorch-platform-gpu" % "1.13.1-1.5.9-SNAPSHOT"
)
```

  </TabItem>
  <TabItem value="scala-cli" label="Scala-CLI">

```scala
//> using repository "sonatype:snapshots"
//> using lib "dev.storch::storch:@VERSION@"
//> using lib "org.bytedeco:pytorch-platform-gpu:1.13.1-1.5.9-SNAPSHOT"
```

  </TabItem>
</Tabs>

This approach should work on any platform with CUDA support (Linux and Windows), but it
`pytorch-platform-gpu`

## Running tensor operations on the GPU

```scala mdoc:invisible
torch.manualSeed(0)
```

You can create tensors directly on the GPU:
```scala mdoc
import torch.Device.{CPU, CUDA}
val device = if torch.cuda.isAvailable then CUDA else CPU
torch.rand(Seq(3,3), device=device)

// Use device index if you have multiple GPUs
torch.rand(Seq(3,3), device=torch.Device(torch.DeviceType.CUDA, 0: Byte))
```
Or move them from the CPU:
```scala mdoc
val cpuTensor = torch.Tensor(Seq(1,2,3))
val gpuTensor = cpuTensor.to(device=device)
```
Tensors stay on their device:
```scala mdoc
cpuTensor + cpuTensor * 100
gpuTensor + gpuTensor * 100
```
Cross-CPU/GPU operations are not allowed: 
```scala mdoc:crash
cpuTensor + gpuTensor
```