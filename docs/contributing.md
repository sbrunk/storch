# How to Contribute

Contributions to Storch are most welcome. Please have a look at the existing
[issues](https://github.com/sbrunk/storch/issues) or don't hesitate to open a new one
to discuss ideas.

## Development setup

You need to have [sbt](https://www.scala-sbt.org/) installed to build Storch from source.

Then clone the git repo and you should be ready to go.
```bash
git clone https://github.com/sbrunk/storch
sbt
```

## Enabling GPU support during development

GPU support is disabled by default. To enable it, set the following parameter inside the sbt shell:

```scala
set ThisBuild / enableGPU := true
```

You can verify if the GPU is working by running the LeNet example.
If it's working, you should see an output like this:

```
sbt examples/runMain LeNetApp
[info] running (fork) LeNetApp 
[info] Using device: Device(CUDA,-1)
...
```

## Edit the documentation

Documentation sources live in the *docs* directory.
We use [mdoc](https://scalameta.org/mdoc/) for typechecked documenation and to embed code output.
The website is rendered by [Laika](https://typelevel.org/Laika/).
To build the documentation locally, you can run the following command:

```bash
sbt ~tlSitePreview
```

Then open http://localhost:4242 in a browser enjoy a live preview while hacking on the docs.

To just build Scaladoc for all modules, you can run

```bash
sbt ~unidoc
```

## Linting

Manually run headerCrate + scalafmt on all files:

```bash
sbt 'headerCreateAll ; scalafmtAll'
```

Add useful git pre-push linting checks:

```bash
cp git-hooks/pre-push-checks .git/hooks/ && chmod +x git-hooks/pre-push-checks
```

## Optional: Install dependencies via nix + devenv

You can use nix and devenv to install your develepment environment, but it's not required.

1. Install [nix](https://nixos.org) package manager

```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

For more info, see https://nixos.org/download.html

2. Install [devenv](https://devenv.sh)

```bash
nix profile install --accept-flake-config github:cachix/devenv/latest
```

For more info, see: https://devenv.sh/getting-started/#installation

3. (Optionally) Install [direnv](https://direnv.net/)

This will load the specific environment variables upon `cd` into the storch folder

```bash
nix profile install 'nixpkgs#direnv'
```

4. Load environment

If you did not install direnv, run the following in the `storch` root folder:

```bash
devenv shell
```

If you installed direnv, just `cd` into storch