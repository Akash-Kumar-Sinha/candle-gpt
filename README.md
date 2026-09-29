# Candle GPT

A lightweight, pure Rust implementation of GPT-2 inference featuring an interactive streaming conversational CLI.

## Downloading Weights

Before running the model, download and export the GPT-2 weights and tokenizer using the included Python script:

```bash
cd gp2-weights
uv sync
uv run main.py
cd ..
```

This will download the pretrained `gpt2` model from Hugging Face and save the exported weights and tokenizer under the `weights/` directory.

## Quick Start

You can use the simple commands provided in the `Makefile`:

- **Start Chat CLI (Default 128 max tokens)**:

  ```bash
  make cli
  ```

- **Start Chat CLI with custom token limit (e.g., 512 tokens)**:
  ```bash
  make cli 512
  ```

## Custom Generation Flags

You can also run directly with `cargo` to tweak sampling parameters:

```bash
cargo run --release -- --max-new-tokens 256 --temperature 0.7 --top-k 40 --top-p 0.9
```

> **Note:** Running the release binary (`cargo run --release`) is significantly faster than `cargo run`.
