package dev.storch.sbt

import sbt._, Keys._

//import StorchSitePlugin.autoImport._
import org.typelevel.sbt.TypelevelSitePlugin
import org.typelevel.sbt.TypelevelSitePlugin.autoImport._
import laika.sbt.LaikaPlugin.autoImport._

import laika.helium.config._
import laika.ast.Path._
import laika.ast._

import laika.ast.LengthUnit._
import laika.ast._
import laika.helium.Helium
import laika.helium.config.Favicon
import laika.helium.config.HeliumIcon
import laika.helium.config.IconLink
import laika.helium.config.ImageLink
import laika.config.{ApiLinks, LinkConfig, ChoiceConfig, SelectionConfig, Selections}

import laika.sbt.LaikaPlugin
import laika.theme.ThemeProvider
import laika.theme.config.{CrossOrigin, ScriptAttributes, StyleAttributes}

object StorchSitePlugin extends AutoPlugin {

  override def requires = TypelevelSitePlugin
  override def projectSettings = Seq(
    laikaConfig := LaikaConfig.defaults.withRawContent
      .withConfigValue(
        LinkConfig.empty
          // .addApiLinks(ApiLinks(baseUri = "http://localhost:4242/api/")
          .addApiLinks(ApiLinks(baseUri = "https://storch.dev/api/"))
      )
      .withConfigValue(
        Selections(
          SelectionConfig(
            "build-tool",
            ChoiceConfig("sbt", "sbt"),
            ChoiceConfig("scala-cli", "Scala CLI")
          ).withSeparateEbooks
        )
      ),
    tlSiteHelium := tlSiteHelium.value.site
      .metadata(
        title = Some("Storch"),
        authors = developers.value.map(_.name),
        language = Some("en"),
        version = Some(version.value.toString)
      )
      .site
      .layout(
        contentWidth = px(860),
        navigationWidth = px(275),
        topBarHeight = px(50),
        defaultBlockSpacing = px(10),
        defaultLineHeight = 1.5,
        anchorPlacement = laika.helium.config.AnchorPlacement.Right
      )
      //        .site
      //        .favIcons(
      //          Favicon.external("https://typelevel.org/img/favicon.png", "32x32", "image/png")
      //        )
      .site
      .topNavigationBar(
        navLinks = Seq(
          IconLink.internal(
            Root / "api" / "index.html",
            HeliumIcon.api,
            options = Styles("svg-link")
          )
          //            IconLink.external("https://discord.gg/XF3CXcMzqD", HeliumIcon.chat),
          //            IconLink.external("https://twitter.com/typelevel", HeliumIcon.twitter)
        )
      )
      .site
      .mainNavigation(
        appendLinks = Seq(
          ThemeNavigationSection(
            "Related Projects",
            TextLink.external("https://pytorch.org/", "PyTorch"),
            TextLink.external("https://github.com/bytedeco/javacpp", "JavaCPP")
          )
        )
      )
      .site
      .landingPage(
        logo = Some(
          Image.internal(Root / "img" / "storch.svg", height = Some(Length(300, LengthUnit.px)))
        ),
        title = Some("Storch"),
        subtitle = Some("GPU Accelerated Deep Learning for Scala 3"),
        license = Some("Apache 2"),
        //          titleLinks = Seq(
        //            VersionMenu.create(unversionedLabel = "Getting Started"),
        //            LinkGroup.create(
        //              IconLink.external("https://github.com/abcdefg/", HeliumIcon.github),
        //              IconLink.external("https://gitter.im/abcdefg/", HeliumIcon.chat),
        //              IconLink.external("https://twitter.com/abcdefg/", HeliumIcon.twitter)
        //            )
        //          ),
        documentationLinks = Seq(
          TextLink.internal(Root / "about.md", "About"),
          TextLink.internal(Root / "installation.md", "Getting Started"),
          TextLink.internal(Root / "api" / "index.html", "API (Scaladoc)")
        ),
        projectLinks = Seq(
          IconLink.external(
            scmInfo.value.fold("https://github.com/sbrunk/storch")(_.browseUrl.toString),
            HeliumIcon.github,
            options = Styles("svg-link")
          )
        ),
        teasers = Seq(
          Teaser(
            "Build Deep Learning Models in Scala",
            """
          |Storch provides GPU accelerated tensor operations, automatic differentiation,
          |and a neural network API for building and training machine learning models.
          |""".stripMargin
          ),
          Teaser(
            "Get the Best of PyTorch & Scala",
            """
          |Storch aims to be close to the original PyTorch API, while still leveraging Scala's powerful type
          |system for safer tensor operations.
          |""".stripMargin
          ),
          Teaser(
            "Powered by LibTorch & JavaCPP",
            """
          |Storch is based on <a href="https://pytorch.org/cppdocs/">LibTorch</a>, the C++ library underlying PyTorch.
          |JVM bindings are provided by <a href="https://github.com/bytedeco/javacpp">JavaCPP</a> for seamless
          |interop with native code & CUDA support.
          |""".stripMargin
          )
        )
      )
      .site
      .internalCSS(Root / "css") // custom styles
      // KaTeX
      .site
      .externalCSS(
        url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
        attributes = StyleAttributes.defaults
          .withIntegrity("sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV")
          .withCrossOrigin(CrossOrigin.Anonymous)
      )
      .site
      .externalJS(
        url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js",
        attributes = ScriptAttributes.defaults.defer
          .withIntegrity("sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8")
          .withCrossOrigin(CrossOrigin.Anonymous)
      )
      .site
      .externalJS(
        url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js",
        attributes = ScriptAttributes.defaults.defer
          .withIntegrity("sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05")
          .withCrossOrigin(CrossOrigin.Anonymous)
      )
      .site
      .internalJS(Root / "js" / "render-katex.js")
  )
}
