# rust-candle-chatbot
RCC is a LLM Terminal Chat - Rust-based Chatbot with Llama 3.2

A high-performance terminal-based chatbot built with Rust, featuring:
- 🚀 **Candle** for fast ML inference
- 💻 **Cursive** for an interactive terminal UI
- 🤖 **Llama 3.2 3B** quantized model (Q4_K_M)
- ⚡ **CUDA** support for GPU acceleration

## Prerequisites

1. **Rust** (latest stable version)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **CUDA** (optional, for GPU acceleration)
   - Install CUDA toolkit 11.8 or 12.x
   - Ensure `nvcc` is in your PATH

## Installation

1. Clone the repository and navigate to the project:
   ```bash
   mkdir llm-terminal-chat
   cd llm-terminal-chat
   ```

2. Create the project files:
   - Save the `Cargo.toml` file
   - Save the `main.rs` file in `src/` directory

3. Build the project:
   ```bash
   cargo build --release
   ```

## Obtaining the Model

### Option 1: Automatic Download
Use the `--download` flag to automatically download the model from HuggingFace:

```bash
cargo run --release -- --download
```

### Option 2: Manual Download
Download the quantized model manually from HuggingFace:

1. Visit: https://huggingface.co/QuantFactory/Llama-3.2-3B-Instruct-GGUF
2. Download: `Llama-3.2-3B-Instruct.Q4_K_M.gguf`
3. Place it in your project directory

### Option 3: Use a Different Model
You can use any GGUF quantized Llama model:

```bash
cargo run --release -- --model-path path/to/your/model.gguf
```

## Running the Chatbot

### Basic Usage
```bash
cargo run --release
```

### With Custom Parameters
```bash
cargo run --release -- \
  --model-path llama-3.2-3b-instruct-q4_k_m.gguf \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-seq-len 4096
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to the GGUF model file | `llama-3.2-3b-instruct-q4_k_m.gguf` |
| `--tokenizer-path` | Path to tokenizer.json | Auto-download from HF |
| `--temperature` | Sampling temperature (0.0-1.0) | `0.7` |
| `--top-p` | Top-p sampling parameter | `0.9` |
| `--max-seq-len` | Maximum sequence length | `2048` |
| `--download` | Download model from HuggingFace | `false` |

## Using the Chat Interface

### Controls
- **Enter**: Send message
- **Ctrl+L**: Clear chat history
- **Ctrl+Q**: Quit application
- **Arrow Keys**: Navigate through chat history
- **Page Up/Down**: Scroll through long conversations

### Features
- 📝 Real-time streaming responses
- 🎨 Syntax-highlighted interface
- 💾 Conversation history
- 🔄 Context-aware responses
- ⚡ GPU acceleration (when available)

## Performance Tips

1. **Use GPU acceleration**: The chatbot automatically uses CUDA if available
2. **Optimize quantization**: Q4_K_M provides good balance between quality and speed
3. **Adjust batch size**: For better throughput with multiple users
4. **Memory usage**: The Q4_K_M model uses approximately 2-3GB of RAM/VRAM

## Troubleshooting

### Model Loading Issues
```bash
# If model fails to load, try re-downloading:
rm llama-3.2-3b-instruct-q4_k_m.gguf
cargo run --release -- --download
```

### CUDA Not Detected
```bash
# Check CUDA installation:
nvcc --version

# Set CUDA path if needed:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Out of Memory
- Reduce `--max-seq-len` parameter
- Use a smaller quantization (Q4_0 instead of Q4_K_M)
- Close other GPU-intensive applications

## Architecture Overview

```
┌─────────────────────────────────────┐
│         Cursive Terminal UI         │
├─────────────────────────────────────┤
│          Chat Management            │
│    • Message History                │
│    • Input Processing               │
│    • Response Formatting            │
├─────────────────────────────────────┤
│         Candle Inference            │
│    • Model Loading (GGUF)           │
│    • Token Generation               │
│    • Sampling (temp, top-p)         │
├─────────────────────────────────────┤
│      Hardware Acceleration          │
│    • CUDA (if available)            │
│    • CPU fallback                   │
└─────────────────────────────────────┘
```

## Customization

### Adding Custom System Prompts
Modify the `generate_response` function in `main.rs`:

```rust
// Add system message at the beginning
formatted_prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
formatted_prompt.push_str("You are a helpful assistant...<|eot_id|>");
```

### Changing the UI Theme
Modify the `create_theme` function to customize colors:

```rust
theme.palette[PaletteColor::Primary] = Color::Light(BaseColor::Magenta);
theme.palette[PaletteColor::Secondary] = Color::Light(BaseColor::Yellow);
```

## License

MIT License - Feel free to modify and distribute as needed.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) - Rust-based ML framework
- [Cursive](https://github.com/gyscos/cursive) - Terminal UI library
- [Meta Llama](https://ai.meta.com/llama/) - Llama model family
- [QuantFactory](https://huggingface.co/QuantFactory) - Quantized model providers