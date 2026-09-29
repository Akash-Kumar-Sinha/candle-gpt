use clap::Parser;
use std::io::{self, BufRead};

mod pipeline;
mod print_statement;

use pipeline::{CandlePipeline, GenerationConfig};
use print_statement::{
    LoadingAnimation, print_banner, print_candle_prompt, print_stream_end, print_stream_token,
    print_user_prompt,
};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "candle-gpt",
    version,
    about = "Candle GPT Interactive Conversational CLI"
)]
pub struct Args {
    #[arg(long, alias = "max-tokens", default_value_t = 128)]
    pub max_new_tokens: usize,

    #[arg(long, default_value_t = 20)]
    pub min_tokens: usize,

    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,

    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    #[arg(long, default_value_t = 1.15)]
    pub repetition_penalty: f32,

    #[arg(long, default_value_t = 2.5)]
    pub eos_bias: f32,

    #[arg(long, default_value_t = 10.0)]
    pub eos_punct_bias: f32,
}

fn main() {
    let args = Args::parse();
    let config = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        min_tokens: args.min_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        eos_bias: args.eos_bias,
        eos_punct_bias: args.eos_punct_bias,
    };

    let mut loading = LoadingAnimation::start("Loading weights");
    let pipeline = match CandlePipeline::new() {
        Ok(p) => {
            loading.stop();
            p
        }
        Err(e) => {
            loading.stop();
            eprintln!("Failed to initialize Candle pipeline: {e}");
            std::process::exit(1);
        }
    };

    run_conversation(&pipeline, &config);
}

fn run_conversation(pipeline: &CandlePipeline, config: &GenerationConfig) {
    print_banner();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print_user_prompt();

        match lines.next() {
            Some(Ok(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    println!("\nGoodbye!");
                    break;
                }

                print_candle_prompt();
                match pipeline.generate_stream_with_config(
                    trimmed,
                    config,
                    |token_text, _token_id| {
                        print_stream_token(token_text);
                    },
                ) {
                    Ok(output) => {
                        print_stream_end(&output);
                        println!();
                    }
                    Err(e) => {
                        eprintln!("\nError: {e}\n");
                    }
                }
            }
            Some(Err(e)) => {
                eprintln!("Error reading input: {e}");
                break;
            }
            None => {
                println!("\nGoodbye!");
                break;
            }
        }
    }
}
