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

import dev.dirs.BaseDirectories
import scala.util.Using
import java.nio.file.Files
import java.net.URL

/** Utilities to download and cache pre-trained model weights. */
object hub:
  private val storchDir = os.Path(BaseDirectories.get().cacheDir) / "storch"
  private val hubDir = storchDir / "hub"
  private val modelDir = hubDir / "checkpoints"

  def loadStateDictFromUrl(url: String): Map[String, Tensor[DType]] =
    os.makeDir.all(modelDir)
    val filename = os.Path(URL(url).getPath).last
    val cachedFile = (modelDir / filename)
    if !os.exists(cachedFile) then
      System.err.println(s"Downloading: $url to $cachedFile")
      Using.resource(URL(url).openStream()) { inputStream =>
        val _ = Files.copy(inputStream, cachedFile.toNIO)
      }
    torch.pickleLoad(cachedFile.toNIO)
