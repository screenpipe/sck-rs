//! Example: List all available windows

use sck_rs::Window;

fn main() {
    // Initialize tracing for debug output
    tracing_subscriber::fmt::init();

    println!("Listing all windows...\n");

    match Window::all() {
        Ok(windows) => {
            println!("Found {} windows:\n", windows.len());

            for window in windows {
                let id = window.id().unwrap_or(0);
                let app = window.app_name().unwrap_or_default();
                let title = window.title().unwrap_or_default();
                let pid = window.pid().unwrap_or(0);
                let width = window.width().unwrap_or(0);
                let height = window.height().unwrap_or(0);
                let x = window.x().unwrap_or(0);
                let y = window.y().unwrap_or(0);
                let minimized = window.is_minimized().unwrap_or(false);

                println!(
                    "[{}] {} - \"{}\"",
                    id, app, title
                );
                println!(
                    "     PID: {} | Size: {}x{} | Position: ({}, {}) | Minimized: {}",
                    pid, width, height, x, y, minimized
                );
                println!();
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("\nMake sure screen recording permission is granted in:");
            eprintln!("System Settings > Privacy & Security > Screen Recording");
        }
    }
}
