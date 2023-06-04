# Contributing to Storch


## Install dependencies via nix + devenv

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

3. [Optionally] Install [direnv](https://direnv.net/)

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


## Linting

### Manually run headerCrate + scalafmt on all files

```bash
sbt 'headerCreateAll ; scalafmtAll'
```

### Add useful git pre-push linting checks

```bash
cp git-hooks/pre-push-checks .git/hooks/ && chmod +x git-hooks/pre-push-checks
```
