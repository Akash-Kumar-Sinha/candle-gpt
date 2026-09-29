use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use crate::pipeline::PipelineOutput;

pub const CYAN: &str = "\x1b[36m";
pub const BOLD_CYAN: &str = "\x1b[1;36m";
pub const BOLD_GREEN: &str = "\x1b[1;32m";
pub const DIM: &str = "\x1b[2m";
pub const RESET: &str = "\x1b[0m";

pub fn print_banner() {
    println!("{CYAN}");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                      ║");
    println!("║                         WELCOME TO                                   ║");
    println!("║                                                                      ║");
    println!("║          ██████╗  █████╗ ███╗   ██╗██████╗ ██╗     ███████╗          ║");
    println!("║         ██╔════╝ ██╔══██╗████╗  ██║██╔══██╗██║     ██╔════╝          ║");
    println!("║         ██║      ███████║██╔██╗ ██║██║  ██║██║     █████╗            ║");
    println!("║         ██║      ██╔══██║██║╚██╗██║██║  ██║██║     ██╔══╝            ║");
    println!("║         ╚██████╗ ██║  ██║██║ ╚████║██████╔╝███████╗███████╗          ║");
    println!("║          ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚══════╝          ║");
    println!("║                                                                      ║");
    println!("║                Candle GPT Interactive Conversation                   ║");
    println!("║                                                                      ║");
    println!("║              Type your message and press Enter to chat               ║");
    println!("║                    Type 'exit' or Ctrl+D to quit                     ║");
    println!("║                                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("{RESET}");
}

pub fn print_user_prompt() {
    print!("{BOLD_GREEN}You>{RESET} ");
    io::stdout().flush().ok();
}

pub fn print_candle_prompt() {
    print!("{BOLD_CYAN}Candle>{RESET} ");
    io::stdout().flush().ok();
}

pub fn print_stream_token(token_text: &str) {
    print!("{token_text}");
    io::stdout().flush().ok();
}

pub fn print_stream_end(output: &PipelineOutput) {
    println!();
    println!("{DIM}  ↳ Input tokens: {:?}{RESET}", output.input_tokens);
    println!(
        "{DIM}  ↳ Generated tokens: {:?}{RESET}",
        output.generated_tokens
    );
    println!("{DIM}  ↳ Complete text: {}{RESET}", output.full_text);
}

#[allow(dead_code)]
pub fn print_output(output: &PipelineOutput) {
    println!("{BOLD_CYAN}Candle>{RESET} {}", output.generated_text);
    println!("{DIM}  ↳ Input tokens: {:?}{RESET}", output.input_tokens);
    println!(
        "{DIM}  ↳ Generated tokens: {:?}{RESET}",
        output.generated_tokens
    );
    println!("{DIM}  ↳ Complete text: {}{RESET}", output.full_text);
}

pub struct LoadingAnimation {
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LoadingAnimation {
    pub fn start(message: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);
        let msg = message.to_string();

        let handle = thread::spawn(move || {
            let frames = [".  ", ".. ", "...", "   "];
            let mut i = 0;
            while running_clone.load(Ordering::Relaxed) {
                print!("\r{BOLD_CYAN}{msg}{RESET}{}", frames[i % frames.len()]);
                io::stdout().flush().ok();
                i += 1;
                thread::sleep(Duration::from_millis(250));
            }
            print!("\r\x1b[2K");
            io::stdout().flush().ok();
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    pub fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.running.store(false, Ordering::Relaxed);
            let _ = handle.join();
        }
    }
}

impl Drop for LoadingAnimation {
    fn drop(&mut self) {
        self.stop();
    }
}
