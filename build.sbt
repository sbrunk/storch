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

val scrImageVersion = "4.0.32"
ThisBuild / scalaVersion := "3.2.2"

ThisBuild / githubWorkflowJavaVersions += JavaSpec.temurin("11")
ThisBuild / githubWorkflowPublishTargetBranches := Seq() // disable publishing until publishing infra is ready

val enableGPU = settingKey[Boolean]("enable or disable GPU support")

ThisBuild / enableGPU := false

lazy val commonSettings = Seq(
  Compile / doc / scalacOptions ++= Seq("-groups", "-snippet-compiler:compile"),
  javaCppPresetLibs ++= Seq(
    (if (enableGPU.value) "pytorch-gpu" else "pytorch") -> "1.13.1",
    /*"mkl" -> "2022.2",*/ "openblas" -> "0.3.21"
  ),
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
  .in(file("site"))
  .enablePlugins(ScalaUnidocPlugin, TypelevelSitePlugin, StorchSitePlugin)
  .settings(commonSettings)
  .settings(
    mdocVariables ++= Map(
      "JAVACPP_VERSION" -> javaCppVersion.value
    ),
    ScalaUnidoc / unidoc / unidocProjectFilter := inAnyProject -- inProjects(examples),
    Laika / sourceDirectories ++= Seq(sourceDirectory.value),
    laikaIncludeAPI := true,
    laikaGenerateAPI / mappings := (ScalaUnidoc / packageDoc / mappings).value
  )
  .dependsOn(vision)

lazy val root = project
  .in(file("."))
  .aggregate(core, vision, examples, docs)
