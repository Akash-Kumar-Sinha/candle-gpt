use std::io::{self, BufRead};
use clap::Parser;

mod pipeline;
mod print_statement;

use pipeline::CandlePipeline;
use print_statement::{print_banner, print_output, print_user_prompt, LoadingAnimation};

#[derive(Parser, Debug)]
#[command(
    name = "candle-gpt",
    version,
    about = "Candle GPT Interactive Conversational CLI"
)]
struct Args {}

fn main() {
    let _args = Args::parse();

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

    run_conversation(&pipeline);
}

fn run_conversation(pipeline: &CandlePipeline) {
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

                match pipeline.process(trimmed) {
                    Ok(output) => {
                        print_output(&output);
                        println!();
                    }
                    Err(e) => {
                        eprintln!("Error: {e}\n");
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
