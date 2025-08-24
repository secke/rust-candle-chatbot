use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn;
use candle_transformers::models::quantized_llama::ModelWeights;
use clap::Parser;
use cursive::{
    event::Event,
    theme::{BaseColor, BorderStyle, Color, PaletteColor, Theme},
    traits::{Nameable, Resizable, Scrollable},
    utils::markup::StyledString,
    views::{
        Button, Dialog, EditView, LinearLayout, Panel,
        TextView, DummyView, OnEventView,
    },
    Cursive, CursiveExt,
};
// use hf_hub::api::tokio::Api;
use rand::Rng;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the quantized model file (GGUF format)
    #[arg(long, default_value = "llama-3.2-3b-instruct-q4_k_m.gguf")]
    model_path: PathBuf,
    
    /// Path to the tokenizer file
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,
    
    /// Maximum sequence length
    #[arg(long, default_value = "2048")]
    max_seq_len: usize,
    
    /// Temperature for sampling
    #[arg(long, default_value = "0.7")]
    temperature: f64,
    
    /// Top-p for sampling
    #[arg(long, default_value = "0.9")]
    top_p: f64,
    
    /// Download model from HuggingFace if not found locally
    #[arg(long)]
    download: bool,
}

#[derive(Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

struct ChatState {
    messages: Vec<ChatMessage>,
    model: Arc<Mutex<Option<ModelWeights>>>,
    tokenizer: Arc<Mutex<Option<Tokenizer>>>,
    device: Device,
    args: Args,
}

impl ChatState {
    fn new(args: Args) -> Result<Self> {
        // Use CPU only - no CUDA in codespace
        let device = Device::Cpu;
        
        println!("Using device: CPU");
        
        Ok(Self {
            messages: Vec::new(),
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            device,
            args,
        })
    }
    
    async fn load_model(&mut self) -> Result<()> {
        // Check if model file exists, download if necessary
        if !self.args.model_path.exists() && self.args.download {
            println!("Downloading model from HuggingFace...");
            self.download_model().await?;
        }
        
        // Load the quantized model
        println!("Loading quantized model from {:?}...", self.args.model_path);
        let mut file = std::fs::File::open(&self.args.model_path)?;
        
        // Read GGUF file content first
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;
        
        // Reset file position
        file = std::fs::File::open(&self.args.model_path)?;
        
        // Load model with content
        let model = ModelWeights::from_gguf(content, &mut file, &self.device)?;
        
        *self.model.lock().unwrap() = Some(model);
        
        // Load tokenizer
        let tokenizer = self.load_tokenizer().await?;
        *self.tokenizer.lock().unwrap() = Some(tokenizer);
        
        Ok(())
    }
    
    async fn download_model(&self) -> Result<()> {
        use hf_hub::api::tokio::ApiBuilder;
        
        let api = ApiBuilder::new()
            .with_progress(true)
            .build()?;
            
        let repo = api.model("QuantFactory/Llama-3.2-3B-Instruct-GGUF".to_string());
        
        // Download the Q4_K_M quantized model
        let filename = "Llama-3.2-3B-Instruct.Q4_K_M.gguf";
        println!("Downloading {}...", filename);
        
        // Download to local path
        let downloaded_path = repo.get(filename).await?;
        
        // Copy to desired location
        std::fs::copy(downloaded_path, &self.args.model_path)?;
        
        println!("Model downloaded successfully!");
        Ok(())
    }
    
    async fn load_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer = if let Some(path) = &self.args.tokenizer_path {
            Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
        } else {
            // Download tokenizer from HuggingFace
            use hf_hub::api::tokio::ApiBuilder;
            
            let api = ApiBuilder::new()
                .with_progress(true)
                .build()?;
                
            let repo = api.model("meta-llama/Llama-3.2-3B-Instruct".to_string());
            
            println!("Downloading tokenizer from HuggingFace...");
            let downloaded_path = repo.get("tokenizer.json").await?;
            
            // Save tokenizer for future use
            let tokenizer_path = PathBuf::from("tokenizer.json");
            std::fs::copy(&downloaded_path, &tokenizer_path)?;
            
            // Load tokenizer from the downloaded file
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to create tokenizer: {}", e))?
        };
        
        Ok(tokenizer)
    }
    
    fn generate_response(&mut self, prompt: &str) -> Result<String> {
        let mut model = self.model.lock().unwrap();
        let tokenizer = self.tokenizer.lock().unwrap();
        
        if model.is_none() || tokenizer.is_none() {
            return Err(anyhow::anyhow!("Model or tokenizer not loaded"));
        }
        
        let model = model.as_mut().unwrap();
        let tokenizer = tokenizer.as_ref().unwrap();
        
        // Format the conversation with Llama3 chat template
        let mut formatted_prompt = String::new();
        formatted_prompt.push_str("<|begin_of_text|>");
        
        for msg in &self.messages {
            formatted_prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.role, msg.content
            ));
        }
        
        formatted_prompt.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            prompt
        ));
        formatted_prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        
        // Tokenize input
        let encoding = tokenizer.encode(formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();
        let mut tokens = input_ids.to_vec();
        
        // Generate response
        let mut generated_tokens = Vec::new();
        let max_new_tokens = 512;
        let eos_token_id = 128009; // <|eot_id|> token
        
        for _ in 0..max_new_tokens {
            // Create input tensor from all tokens
            let input_tensor = Tensor::new(tokens.as_slice(), &self.device)?
                .unsqueeze(0)?;
            
            // Get model output
            let logits = model.forward(&input_tensor, tokens.len() - 1)?;
            
            // Sample next token
            let next_token = self.sample_token(&logits)?;
            generated_tokens.push(next_token);
            tokens.push(next_token);
            
            // Check for end token
            if next_token == eos_token_id {
                break;
            }
        }
        
        // Decode response
        let response = tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
        
        Ok(response)
    }
    
    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        // Get the last token's logits (handle different tensor shapes)
        let logits = if logits.dims().len() > 1 {
            logits.squeeze(0)?
        } else {
            logits.clone()
        };
        
        // If we have a sequence dimension, take the last token
        let logits = if logits.dims().len() > 1 {
            let seq_len = logits.dims()[0];
            logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        } else {
            logits
        };
        
        let logits = logits.to_dtype(DType::F32)?;
        
        // Apply temperature
        let logits = (&logits / self.args.temperature)?;
        
        // Apply softmax
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        
        // Apply top-p sampling
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed_probs.len();
        
        for (i, (_, p)) in indexed_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.args.top_p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Sample from the filtered distribution
        let filtered_probs = &indexed_probs[..cutoff_idx];
        let sum: f32 = filtered_probs.iter().map(|(_, p)| p).sum();
        let normalized: Vec<f32> = filtered_probs.iter().map(|(_, p)| p / sum).collect();
        
        // Random sampling
        let mut rng = rand::thread_rng();
        let sample = rng.gen::<f32>();
        
        let mut cumsum = 0.0;
        for (i, p) in normalized.iter().enumerate() {
            cumsum += p;
            if cumsum >= sample {
                return Ok(filtered_probs[i].0 as u32);
            }
        }
        
        Ok(filtered_probs.last().unwrap().0 as u32)
    }
}

fn create_theme() -> Theme {
    let mut theme = Theme::default();
    
    theme.palette[PaletteColor::Background] = Color::Dark(BaseColor::Black);
    theme.palette[PaletteColor::View] = Color::Dark(BaseColor::Black);
    theme.palette[PaletteColor::Primary] = Color::Light(BaseColor::Cyan);
    theme.palette[PaletteColor::Secondary] = Color::Light(BaseColor::Blue);
    theme.palette[PaletteColor::Tertiary] = Color::Light(BaseColor::Green);
    theme.palette[PaletteColor::TitlePrimary] = Color::Light(BaseColor::Yellow);
    theme.palette[PaletteColor::TitleSecondary] = Color::Light(BaseColor::Green);
    theme.palette[PaletteColor::Highlight] = Color::Dark(BaseColor::Blue);
    theme.palette[PaletteColor::HighlightInactive] = Color::Dark(BaseColor::Black);
    
    theme.borders = BorderStyle::Simple;
    
    theme
}

fn build_ui(siv: &mut Cursive, state: Arc<Mutex<ChatState>>) {
    siv.set_theme(create_theme());
    
    let chat_view = TextView::new("Welcome to LLM Terminal Chat!\n\nType your message below and press Enter to send.\n")
        .with_name("chat_view")
        .scrollable()
        .scroll_strategy(cursive::view::scroll::ScrollStrategy::StickToBottom)
        .min_height(20);
    
    let input_view = EditView::new()
        .on_submit(move |s, text| {
            if text.is_empty() {
                return;
            }
            
            let state = state.clone();
            let user_msg = text.to_string();
            
            // Add user message to chat
            s.call_on_name("chat_view", |view: &mut TextView| {
                let mut content = StyledString::new();
                content.append_styled("\n[You]: ", Color::Light(BaseColor::Green));
                content.append_plain(&user_msg);
                view.append(content);
            });
            
            // Clear input
            s.call_on_name("input_field", |view: &mut EditView| {
                view.set_content("");
            });
            
            // Generate response in background
            let cb_sink = s.cb_sink().clone();
            std::thread::spawn(move || {
                let mut state = state.lock().unwrap();
                
                // Add user message to history
                state.messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: user_msg.clone(),
                });
                
                // Generate response
                match state.generate_response(&user_msg) {
                    Ok(response) => {
                        // Add assistant message to history
                        state.messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: response.clone(),
                        });
                        
                        cb_sink.send(Box::new(move |s| {
                            s.call_on_name("chat_view", |view: &mut TextView| {
                                let mut content = StyledString::new();
                                content.append_styled("\n[Assistant]: ", Color::Light(BaseColor::Cyan));
                                content.append_plain(&response);
                                view.append(content);
                            });
                        })).unwrap();
                    }
                    Err(e) => {
                        cb_sink.send(Box::new(move |s| {
                            s.call_on_name("chat_view", |view: &mut TextView| {
                                let mut content = StyledString::new();
                                content.append_styled("\n[Error]: ", Color::Light(BaseColor::Red));
                                content.append_plain(&format!("{}", e));
                                view.append(content);
                            });
                        })).unwrap();
                    }
                }
            });
        })
        .with_name("input_field")
        .full_width();
    
    let input_panel = Panel::new(input_view)
        .title("Input")
        .title_position(cursive::align::HAlign::Left);
    
    let layout = LinearLayout::vertical()
        .child(Panel::new(chat_view)
            .title("Chat History")
            .title_position(cursive::align::HAlign::Center)
            .full_height())
        .child(DummyView.fixed_height(1))
        .child(input_panel)
        .child(DummyView.fixed_height(1))
        .child(LinearLayout::horizontal()
            .child(TextView::new("Commands: "))
            .child(Button::new("Clear (Ctrl+L)", |s| {
                s.call_on_name("chat_view", |view: &mut TextView| {
                    view.set_content("");
                });
            }))
            .child(TextView::new(" | "))
            .child(Button::new("Quit (Ctrl+Q)", |s| s.quit())));
    
    // Add keyboard shortcuts
    let layout = OnEventView::new(layout)
        .on_event(Event::CtrlChar('l'), |s| {
            s.call_on_name("chat_view", |view: &mut TextView| {
                view.set_content("Chat cleared.\n");
            });
        })
        .on_event(Event::CtrlChar('q'), |s| {
            s.quit();
        });
    
    siv.add_fullscreen_layer(layout);
    
    // Show loading dialog
    siv.add_layer(
        Dialog::around(TextView::new("Loading model...\nThis may take a few minutes on first run."))
            .title("Initializing")
            .with_name("loading_dialog")
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();
    
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Create chat state
    let chat_state = ChatState::new(args)?;
    
    // Load model in background
    let state_arc = Arc::new(Mutex::new(chat_state));
    let state_clone = state_arc.clone();
    
    // Create Cursive app
    let mut siv = Cursive::default();
    build_ui(&mut siv, state_arc.clone());
    
    // Load model in background
    let cb_sink = siv.cb_sink().clone();
    std::thread::spawn(move || {
        // Use blocking runtime for async operations in thread
        let runtime = tokio::runtime::Runtime::new().unwrap();
        
        let load_result = runtime.block_on(async {
            let mut state = state_clone.lock().unwrap();
            state.load_model().await
        });
        
        match load_result {
            Ok(_) => {
                cb_sink.send(Box::new(|s| {
                    s.pop_layer(); // Remove loading dialog
                    s.call_on_name("chat_view", |view: &mut TextView| {
                        let mut content = StyledString::new();
                        content.append_styled("Model loaded successfully!\n\n", Color::Light(BaseColor::Green));
                        content.append_plain("You can now start chatting. Type your message and press Enter.\n");
                        view.set_content(content);
                    });
                })).unwrap();
            }
            Err(e) => {
                cb_sink.send(Box::new(move |s| {
                    s.pop_layer(); // Remove loading dialog
                    s.add_layer(
                        Dialog::text(format!("Failed to load model: {}\n\nPlease check the model path or use --download flag.", e))
                            .title("Error")
                            .button("Quit", |s| s.quit())
                    );
                })).unwrap();
            }
        }
    });
    
    // Run the UI
    siv.run();
    
    Ok(())
}