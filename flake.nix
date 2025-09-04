{
  description = "revolutionary new technology that turns any image into obama";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      ...
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [
                rust-overlay.overlays.default
                self.overlays.default
              ];
            };
            systemStr = system;
          }
        );
    in
    {
      overlays.default = final: prev: {
        rustToolchain =
          let
            rust = prev.rust-bin;
          in
          if builtins.pathExists ./rust-toolchain.toml then
            rust.fromRustupToolchainFile ./rust-toolchain.toml
          else if builtins.pathExists ./rust-toolchain then
            rust.fromRustupToolchainFile ./rust-toolchain
          else
            rust.stable.latest.default.override {
              extensions = [
                "rust-src"
                "rustfmt"
              ];
            };
      };

      packages = forEachSupportedSystem (
        { pkgs, systemStr }:
        {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "obamify";
            version = "1.1";

            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            nativeBuildInputs = with pkgs; [
              pkg-config
              makeWrapper
            ];
            buildInputs = [
              pkgs.openssl
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isLinux (
              with pkgs;
              [
                wayland
                libxkbcommon
                xorg.libX11
                xorg.libXcursor
                xorg.libXrandr
                xorg.libXi
                vulkan-loader
                libglvnd
                mesa
                egl-wayland
              ]
            );

            postFixup = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              wrapProgram $out/bin/obamify \
                --set-default WINIT_UNIX_BACKEND wayland \
                --set-default WGPU_BACKEND vulkan \
                --set LD_LIBRARY_PATH ${
                  pkgs.lib.makeLibraryPath [
                    pkgs.wayland
                    pkgs.libxkbcommon
                    pkgs.xorg.libX11
                    pkgs.xorg.libXcursor
                    pkgs.xorg.libXrandr
                    pkgs.xorg.libXi
                    pkgs.vulkan-loader
                    pkgs.libglvnd
                    pkgs.mesa
                    pkgs.egl-wayland
                  ]
                }
            '';

            enableParallelBuild = true;

            meta = {
              description = "revolutionary new technology that turns any image into obama";
              homepage = "htpps://github/Spu7Nix/obamify";
              license = pkgs.lib.licenses.mit;
              mainProgram = "obamify";
            };
          };
        }
      );

      apps = forEachSupportedSystem (
        { pkgs, systemStr }:
        {
          default = {
            type = "app";
            program = nixpkgs.lib.getExe self.packages.${systemStr}.default;
          };
        }
      );

      devShells = forEachSupportedSystem (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustToolchain
                openssl
                pkg-config
                cargo-deny
                cargo-edit
                cargo-watch
                rust-analyzer
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
                wayland
                libxkbcommon
                xorg.libX11
                xorg.libXcursor
                xorg.libXrandr
                xorg.libXi
              ];

            env = {
              RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
            };
          };
        }
      );
    };
}
