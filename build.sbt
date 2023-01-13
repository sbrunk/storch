val scrImageVersion = "4.0.32"
ThisBuild / scalaVersion := "3.2.1"

val enableGPU = settingKey[Boolean]("enable or disable GPU support")

lazy val commonSettings = Seq(
  enableGPU := true,
  scalaVersion := "3.2.1",
  organization := "dev.storch",
  version := "0.1.0-SNAPSHOT",
  Compile / doc / scalacOptions ++= Seq("-groups", "-snippet-compiler:compile"),
  javaCppPresetLibs ++= Seq((if (enableGPU.value) "pytorch-gpu" else "pytorch") -> "1.13.1", "mkl" -> "2022.2", "openblas" -> "0.3.21"),
  javaCppVersion := "1.5.9-SNAPSHOT",
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
)

lazy val core = project
  .in(file("core"))
  .settings(commonSettings)
  .settings(
    name := "storch",
    fork := true,
    Test / fork := true,
    libraryDependencies ++= Seq(
      "org.bytedeco" % "pytorch" % "1.13.1-1.5.9-SNAPSHOT",
      "org.typelevel" %% "spire" % "0.18.0",
      "com.lihaoyi" %% "sourcecode" % "0.3.0",
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.scalameta" %% "munit-scalacheck" % "0.7.29" % Test
    )
  )

lazy val vision = project
  .in(file("vision"))
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.sksamuel.scrimage" % "scrimage-core" % scrImageVersion,
      "com.sksamuel.scrimage" % "scrimage-webp" % scrImageVersion
    )
  )
  .dependsOn(core)


lazy val examples = project
  .in(file("examples"))
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "os-lib" % "0.9.0",
      "me.tongfei" % "progressbar" % "0.9.5"
    )
  )
  .dependsOn(vision)

lazy val docs = project
  .in(file("storch-docs"))
  .dependsOn(vision)
  .enablePlugins(MdocPlugin, DocusaurusPlugin)
  .settings(commonSettings)
  .settings(
    mdocVariables := Map(
      "VERSION" -> version.value
    )
  )