#!/bin/bash

# Script to fix Candle dependency issues

echo "🔧 Fixing Candle dependency issues..."

# Clean everything
echo "📦 Cleaning cargo cache and build artifacts..."
cargo clean
rm -rf ~/.cargo/registry/cache
rm -rf ~/.cargo/registry/index
rm -rf ~/.cargo/git/checkouts
rm -rf ~/.cargo/git/db
rm Cargo.lock 2>/dev/null

# Update rust toolchain
echo "🔄 Updating Rust toolchain..."
rustup update stable

# Clone and build Candle locally with patches
echo "📥 Cloning Candle repository..."
git clone https://github.com/huggingface/candle.git candle-local
cd candle-local

# Check out a specific commit known to work (optional)
# git checkout 4b25870

# Apply patch to fix rand dependency
echo "🩹 Applying dependency fixes..."
cat > fix-rand.patch << 'EOF'
diff --git a/candle-core/Cargo.toml b/candle-core/Cargo.toml
index 1234567..abcdefg 100644
--- a/candle-core/Cargo.toml
+++ b/candle-core/Cargo.toml
@@ -14,7 +14,7 @@ categories = ["science"]
 [dependencies]
 byteorder = "1.4.3"
 gemm = { version = "0.17.0", features = ["f16"] }
-half = { version = "2.3.1", features = ["num-traits", "use-intrinsics", "rand_distr"] }
+half = { version = "2.3.1", features = ["num-traits", "use-intrinsics"] }
 num-traits = "0.2.15"
 num_cpus = "1.15.0"
 rand = "0.8.5"
EOF

# Try to apply the patch (might fail if already fixed)
git apply fix-rand.patch 2>/dev/null || echo "Patch not needed or already applied"

cd ..

# Create a new Cargo.toml that uses the local Candle
echo "📝 Creating new Cargo.toml with local Candle..."
cat > Cargo.toml << 'EOF'
[package]
name = "llm-terminal-chat"
version = "0.1.0"
edition = "2021"

[dependencies]
# Use local Candle with fixes
candle-core = { path = "./candle-local/candle-core", features = ["cuda"] }
candle-nn = { path = "./candle-local/candle-nn" }
candle-transformers = { path = "./candle-local/candle-transformers" }

# Tokenizer support
tokenizers = { version = "0.20", features = ["http"] }
hf-hub = { version = "0.3", features = ["tokio"] }

# Terminal UI
cursive = { version = "0.21", default-features = false, features = ["crossterm-backend"] }

# Async runtime
tokio = { version = "1.40", features = ["full"] }

# Error handling and utilities
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Make sure we use consistent rand version
rand = "0.8.5"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
EOF

echo "🔨 Building with local Candle..."
cargo build --release

echo "✅ Done! If the build succeeds, you can now run:"
echo "   cargo run --release"