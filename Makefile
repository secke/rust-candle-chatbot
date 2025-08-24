.PHONY: run build clean

run:
	cargo run --release

run_download:
	cargo run --release -- --download

build:
	cargo build --release

clean:
	cargo clean && rm -rf target/ candle-local/ Cargo.lock