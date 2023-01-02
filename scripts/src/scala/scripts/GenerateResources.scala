package scripts

import bleep.{BleepScript, Commands, Started}

import java.nio.file.Files

object GenerateResources extends BleepScript("GenerateResources") {
  def run(started: Started, commands: Commands, args: List[String]): Unit = {
    started.logger.error("This script is a placeholder! You'll need to replace the contents with code which actually generates the files you want")
  }
}
