{ pkgs, lib, ... }:

{
  packages = with pkgs; [
    # tooling outside Python itself
    just
  ];

  languages.python = {
    enable = true;

    # Interpreter comes from Nix, so it is patched to work on NixOS.
    # Leave unset to use the nixpkgs default, or pin an exact version
    # (requires the nixpkgs-python input above):
    version = "3.12";

    uv = {
      enable = true;
      # Runs `uv sync` automatically when you enter the shell,
      # installing everything from pyproject.toml/uv.lock into .venv:
      sync.enable = true;
      # sync.arguments = [ "--group" "dev" ];  # optional: sync only some groups
    };

    # Native C libraries that wheels (numpy, psycopg, ortools, torch, ...)
    # expect. Each becomes part of LD_LIBRARY_PATH:
    libraries = with pkgs; [
      stdenv.cc.cc   # libstdc++.so.6 -- fixes the classic ImportError
      zlib
      # add as needed per project:
      # libGL
      # glib
      # postgresql  # for psycopg
      # "/run/opengl-driver"  # GPU stuff
    ];
  };

  enterShell = ''
    # Belt-and-braces: ensure uv also sees the libraries
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
  '';

  # Example git hooks (optional):
  git-hooks.hooks = {
    ruff.enable = true;
    ruff-format.enable = true;
  };
}