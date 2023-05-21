{ pkgs, inputs, ... }:

let
  packages = if pkgs.stdenv.isDarwin
             then inputs.nixpkgs.legacyPackages.x86_64-darwin
             else pkgs;
in
{
  packages = with packages; [
    sbt
  ];

  scripts.hello.exec = "echo ---STORCH---";

  enterShell = ''
    hello
  '';
}
