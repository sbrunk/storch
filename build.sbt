val scala3Version = "3.1.2"

lazy val root = project
  .in(file("."))
  .settings(
    name := "storch",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies ++= Seq(
      "org.bytedeco" % "pytorch-platform" % "1.10.2-1.5.7",
      "org.bytedeco" % "mkl-platform" % "2022.0-1.5.7",
      "org.scalameta" %% "munit" % "0.7.29" % Test
    )
  )
