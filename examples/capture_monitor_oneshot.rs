//! Example: One-shot monitor capture via SCScreenshotManager
//!
//! Unlike `capture_monitor.rs` (which uses the persistent SCStream path),
//! this example uses `capture_image_oneshot()` — WindowServer / replayd
//! wakes only for the frame requested, then releases the capture path.
//!
//! Right primitive for event-driven consumers (capture on app switch,
//! click, typing pause, etc.) where capture rate is well below the
//! persistent stream's frame interval.

use sck_rs::Monitor;

fn main() {
    tracing_subscriber::fmt::init();

    let monitors = match Monitor::all() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("\nMake sure screen recording permission is granted.");
            return;
        }
    };

    println!("Capturing {} monitor(s) via one-shot...\n", monitors.len());

    for monitor in monitors {
        let id = monitor.id();
        let name = monitor.name().to_string();

        let started = std::time::Instant::now();
        match monitor.capture_image_oneshot() {
            Ok(image) => {
                let elapsed = started.elapsed();
                let filename = format!("monitor_oneshot_{}.png", id);
                match image.save(&filename) {
                    Ok(_) => println!(
                        "[{}] {} — {}x{} in {:?} → {}",
                        id,
                        name,
                        image.width(),
                        image.height(),
                        elapsed,
                        filename
                    ),
                    Err(e) => println!("[{}] save failed: {}", id, e),
                }
            }
            Err(e) => println!("[{}] capture failed: {}", id, e),
        }
    }
}
