import sbt._

import Keys._
import MdocPlugin.autoImport._
import LaikaPlugin.autoImport._

ThisBuild / tlBaseVersion := "0.0" // your current series x.y

ThisBuild / organization := "dev.storch"
ThisBuild / organizationName := "storch.dev"
ThisBuild / startYear := Some(2022)
ThisBuild / licenses := Seq(License.Apache2)
ThisBuild / developers := List(
  // your GitHub handle and name
  tlGitHubDev("sbrunk", "Sören Brunk")
)

// publish to s01.oss.sonatype.org (set to true to publish to oss.sonatype.org instead)
ThisBuild / tlSonatypeUseLegacyHost := false

// publish website from this branch
ThisBuild / tlSitePublishBranch := Some("main")

ThisBuild / apiURL := Some(new URL("https://storch.dev/api/"))

val scrImageVersion = "4.0.34"
val pytorchVersion = "2.0.1"
val cudaVersion = "12.1-8.9"
val openblasVersion = "0.3.23"
val mklVersion = "2023.1"
ThisBuild / scalaVersion := "3.3.0"
ThisBuild / javaCppVersion := "1.5.9"

ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("11"))

val enableGPU = settingKey[Boolean]("enable or disable GPU support")

ThisBuild / enableGPU := false

lazy val commonSettings = Seq(
  Compile / doc / scalacOptions ++= Seq("-groups", "-snippet-compiler:compile"),
  javaCppVersion := (ThisBuild / javaCppVersion).value,
  javaCppPlatform := Seq(),
  resolvers ++= Resolver.sonatypeOssRepos("snapshots")
  // This is a hack to avoid depending on the native libs when publishing
  // but conveniently have them on the classpath during development.
  // There's probably a cleaner way to do this.
) ++ tlReplaceCommandAlias(
  "tlReleaseLocal",
  List(
    "reload",
    "project /",
    "set core / javaCppPlatform := Seq()",
    "set core / javaCppPresetLibs := Seq()",
    "+publishLocal"
  ).mkString("; ", "; ", "")
) ++ tlReplaceCommandAlias(
  "tlRelease",
  List(
    "reload",
    "project /",
    "set core / javaCppPlatform := Seq()",
    "set core / javaCppPresetLibs := Seq()",
    "+mimaReportBinaryIssues",
    "+publish",
    "tlSonatypeBundleReleaseIfRelevant"
  ).mkString("; ", "; ", "")
)

lazy val core = project
  .in(file("core"))
  .settings(commonSettings)
  .settings(
    javaCppPresetLibs ++= Seq(
      (if (enableGPU.value) "pytorch-gpu" else "pytorch") -> pytorchVersion,
      "mkl" -> mklVersion,
      "openblas" -> openblasVersion
    ) ++ (if (enableGPU.value) Seq("cuda-redist" -> "12.1-8.9") else Seq()),
    javaCppPlatform := org.bytedeco.sbt.javacpp.Platform.current,
    fork := true,
    Test / fork := true,
    libraryDependencies ++= Seq(
      "org.bytedeco" % "pytorch" % s"$pytorchVersion-${javaCppVersion.value}",
      "org.typelevel" %% "spire" % "0.18.0",
      "org.typelevel" %% "shapeless3-typeable" % "3.3.0",
      "com.lihaoyi" %% "os-lib" % "0.9.1",
      "com.lihaoyi" %% "sourcecode" % "0.3.0",
      "dev.dirs" % "directories" % "26",
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
      "com.sksamuel.scrimage" % "scrimage-webp" % scrImageVersion,
      "org.scalameta" %% "munit" % "0.7.29" % Test
    )
  )
  .dependsOn(core)

lazy val examples = project
  .in(file("examples"))
  .enablePlugins(NoPublishPlugin)
  .settings(commonSettings)
  .settings(
    fork := true,
    libraryDependencies ++= Seq(
      "me.tongfei" % "progressbar" % "0.9.5",
      "com.github.alexarchambault" %% "case-app" % "2.1.0-M24",
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(vision)

lazy val docs = project
  .in(file("site"))
  .enablePlugins(ScalaUnidocPlugin, TypelevelSitePlugin, StorchSitePlugin)
  .settings(commonSettings)
  .settings(
    mdocVariables ++= Map(
      "JAVACPP_VERSION" -> javaCppVersion.value,
      "PYTORCH_VERSION" -> pytorchVersion,
      "OPENBLAS_VERSION" -> openblasVersion,
      "MKL_VERSION" -> mklVersion,
      "CUDA_VERSION" -> cudaVersion
    ),
    ScalaUnidoc / unidoc / unidocProjectFilter := inAnyProject -- inProjects(examples),
    Laika / sourceDirectories ++= Seq(sourceDirectory.value),
    laikaIncludeAPI := true,
    laikaGenerateAPI / mappings := (ScalaUnidoc / packageDoc / mappings).value
  )
  .dependsOn(vision)

lazy val root = project
  .enablePlugins(NoPublishPlugin)
  .in(file("."))
  .aggregate(core, vision, examples, docs)
  .settings(
    javaCppVersion := (ThisBuild / javaCppVersion).value
  )
