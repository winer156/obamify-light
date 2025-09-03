# obamify
revolutionary new technology that turns any image into obama

![example](example.gif)

# How to use

**Use the ui at the top of the window to control the animation, choose between saved transformations, and generate new ones.** All your transformations are saved in the `presets` folder next to the executable. I have no idea why you would ever want to do this, but if you want to transform your images to something other than obama, you can change the `target.png` and `weights.png` files in the same directory.

> `weights.png` is a grayscale image that decides how much importance is given to that pixel being accurate in the final image.
> `target.png` is the image that you want to transform your source image into.
> These images need to be the same size, square, and if you make them much larger than 128x128 pixels the result might take hours or even days to generate.

# Installations

Install the latest version in [releases](https://github.com/Spu7Nix/obamify/releases). Unzip and run the .exe file inside!

### Building from source

1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Run `cargo run --release` in the project folder

# How it works

magic

# Contributing

Here are some ideas for features to implement if you're interested:
- Faster algorithms for calculating the image transformation
- Better user experience with saving/loading presets
- Building for web/WASM

Feel free to make an issue or a pull request if you have any ideas :)