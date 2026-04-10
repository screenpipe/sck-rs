//! Persistent SCStream manager — reuses a single stream per monitor.
//!
//! Instead of creating/destroying ScreenCaptureKit objects on every capture,
//! this module maintains a persistent `sc::Stream` per display that delivers
//! frames via a callback. `capture()` simply returns the latest buffered frame.
//!
//! Benefits:
//! - No per-capture `ShareableContent::current()` enumeration
//! - No per-capture `ContentFilter` / `StreamCfg` allocation
//! - `minimumFrameInterval` is respected by macOS (OS-level throttle)
//! - Significantly lower CPU overhead at high capture rates

use cidre::{
    arc, cm, cv, define_obj_type, dispatch, ns, objc, sc,
    sc::stream::{Output, OutputImpl},
};
use image::RgbaImage;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::capture::{get_shareable_content, safe_image_buf_to_rgba};
use crate::error::{XCapError, XCapResult};

// ── Frame receiver (ObjC callback) ─────────────────────────────────

#[repr(C)]
struct FrameReceiverInner {
    latest_frame: Arc<Mutex<Option<RgbaImage>>>,
    frame_notify: Arc<tokio::sync::Notify>,
}

define_obj_type!(
    FrameReceiver + OutputImpl,
    FrameReceiverInner,
    FRAME_RECEIVER_CLS
);

impl Output for FrameReceiver {}

#[objc::add_methods]
impl OutputImpl for FrameReceiver {
    extern "C" fn impl_stream_did_output_sample_buf(
        &mut self,
        _cmd: Option<&cidre::objc::Sel>,
        _stream: &sc::Stream,
        sample_buf: &mut cm::SampleBuf,
        kind: sc::OutputType,
    ) {
        if kind != sc::OutputType::Screen {
            return;
        }

        // Extract image buffer from sample
        let Some(mut image_buf) = sample_buf.image_buf().map(|b| b.retained()) else {
            return;
        };

        // Convert BGRA → RGBA
        let image = match safe_image_buf_to_rgba(&mut image_buf) {
            Ok(img) => img,
            Err(e) => {
                debug!("stream frame conversion failed: {}", e);
                return;
            }
        };

        // Store and notify
        let inner = self.inner_mut();
        if let Ok(mut guard) = inner.latest_frame.lock() {
            *guard = Some(image);
        }
        inner.frame_notify.notify_waiters();
    }
}

// ── Per-monitor stream ─────────────────────────────────────────────

struct MonitorStream {
    _stream: arc::R<sc::Stream>,
    _output: arc::R<FrameReceiver>,
    _queue: arc::R<dispatch::Queue>,
    latest_frame: Arc<Mutex<Option<RgbaImage>>>,
    frame_notify: Arc<tokio::sync::Notify>,
    width: u32,
    height: u32,
    /// SCK window IDs currently excluded from this stream's ContentFilter.
    /// When this changes, the stream must be recreated.
    excluded_window_ids: Vec<u32>,
}

impl MonitorStream {
    fn new(
        sc_display: &sc::Display,
        content: &sc::ShareableContent,
        width: u32,
        height: u32,
        fps: i32,
        excluded_window_ids: &[u32],
    ) -> XCapResult<Self> {
        let latest_frame = Arc::new(Mutex::new(None));
        let frame_notify = Arc::new(tokio::sync::Notify::new());

        let inner = FrameReceiverInner {
            latest_frame: latest_frame.clone(),
            frame_notify: frame_notify.clone(),
        };
        let output = FrameReceiver::with(inner);

        let queue = dispatch::Queue::serial_with_ar_pool();

        // Configure stream
        let mut cfg = sc::StreamCfg::new();
        cfg.set_width(width as usize);
        cfg.set_height(height as usize);
        cfg.set_pixel_format(cv::PixelFormat::_32_BGRA);
        cfg.set_shows_cursor(true);
        cfg.set_scales_to_fit(false);
        cfg.set_minimum_frame_interval(cm::Time::new(1, fps));

        // Build exclusion filter — excluded windows are not rendered by the OS
        let filter = if excluded_window_ids.is_empty() {
            let empty = ns::Array::new();
            sc::ContentFilter::with_display_excluding_windows(sc_display, &empty)
        } else {
            let sc_windows = content.windows();
            let mut to_exclude: Vec<&sc::Window> = Vec::new();
            for w in sc_windows.iter() {
                if excluded_window_ids.contains(&w.id()) {
                    to_exclude.push(w);
                }
            }
            if to_exclude.is_empty() {
                let empty = ns::Array::new();
                sc::ContentFilter::with_display_excluding_windows(sc_display, &empty)
            } else {
                debug!(
                    "persistent stream: excluding {} window(s) for display {}",
                    to_exclude.len(),
                    sc_display.display_id().0
                );
                let arr = ns::Array::from_slice(&to_exclude);
                sc::ContentFilter::with_display_excluding_windows(sc_display, &arr)
            }
        };

        let stream = sc::Stream::new(&filter, &cfg);

        stream
            .add_stream_output(output.as_ref(), sc::OutputType::Screen, Some(&queue))
            .map_err(|e| {
                XCapError::capture_failed(format!("failed to add stream output: {:?}", e))
            })?;

        // Start the stream
        let stream_clone = stream.retained();
        crate::capture::run_in_thread(move || {
            crate::capture::block_on(async { stream_clone.start().await })
        })
        .map_err(|e| XCapError::capture_failed(format!("failed to start stream: {}", e)))?
        .map_err(|e| XCapError::capture_failed(format!("stream start error: {:?}", e)))?;

        info!(
            "persistent SCK stream started for display {} ({}x{}, {}fps, {} excluded)",
            sc_display.display_id().0,
            width,
            height,
            fps,
            excluded_window_ids.len()
        );

        Ok(Self {
            _stream: stream,
            _output: output,
            _queue: queue,
            latest_frame,
            frame_notify,
            width,
            height,
            excluded_window_ids: excluded_window_ids.to_vec(),
        })
    }

    fn latest_frame(&self) -> Option<RgbaImage> {
        self.latest_frame.lock().ok()?.clone()
    }

    async fn wait_first_frame(&self, timeout: Duration) -> XCapResult<RgbaImage> {
        tokio::select! {
            _ = async {
                loop {
                    self.frame_notify.notified().await;
                    if self.latest_frame.lock().ok().map(|g| g.is_some()).unwrap_or(false) {
                        break;
                    }
                }
            } => {},
            _ = tokio::time::sleep(timeout) => {
                return Err(XCapError::capture_failed(
                    "timeout waiting for first stream frame"
                ));
            }
        }

        self.latest_frame()
            .ok_or_else(|| XCapError::capture_failed("no frame after notify"))
    }
}

impl Drop for MonitorStream {
    fn drop(&mut self) {
        debug!("stopping persistent SCK stream");
        let stream = self._stream.retained();
        let _ = crate::capture::run_in_thread(move || {
            crate::capture::block_on(async {
                let _ = stream.stop().await;
            })
        });
    }
}

// ── Stream manager (singleton) ─────────────────────────────────────

struct StreamManager {
    streams: Mutex<HashMap<u32, MonitorStream>>,
}

static MANAGER: Lazy<StreamManager> = Lazy::new(|| StreamManager {
    streams: Mutex::new(HashMap::new()),
});

impl StreamManager {
    /// Get the latest frame for a monitor, creating the stream if needed.
    ///
    /// On first call for a monitor, creates a persistent SCStream and waits
    /// up to 3s for the first frame. Subsequent calls return the latest
    /// buffered frame immediately.
    ///
    /// If `excluded_window_ids` changes from the current stream's exclusion
    /// list, the stream is invalidated and recreated with the new filter.
    pub async fn capture(
        monitor_id: u32,
        width: u32,
        height: u32,
        excluded_window_ids: &[u32],
    ) -> XCapResult<RgbaImage> {
        // Fast path: stream exists, matches params, and has a frame
        {
            let streams = MANAGER
                .streams
                .lock()
                .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
            if let Some(ms) = streams.get(&monitor_id) {
                if ms.width == width
                    && ms.height == height
                    && ms.excluded_window_ids == excluded_window_ids
                {
                    if let Some(frame) = ms.latest_frame() {
                        return Ok(frame);
                    }
                    // Stream exists but no frame yet — wait below
                } else {
                    debug!(
                        "stream params changed for display {} (res or exclusions), recreating",
                        monitor_id
                    );
                }
            }
        }

        // Slow path: create or recreate stream
        Self::ensure_stream(monitor_id, width, height, excluded_window_ids)?;

        // Wait for first frame
        let streams = MANAGER
            .streams
            .lock()
            .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
        if let Some(ms) = streams.get(&monitor_id) {
            let notify = ms.frame_notify.clone();
            let latest = ms.latest_frame.clone();
            drop(streams); // release lock before await

            // Wait for the frame callback to deliver
            tokio::select! {
                _ = async {
                    loop {
                        notify.notified().await;
                        if latest.lock().ok().map(|g| g.is_some()).unwrap_or(false) {
                            break;
                        }
                    }
                } => {},
                _ = tokio::time::sleep(Duration::from_secs(3)) => {
                    return Err(XCapError::capture_failed(
                        "timeout waiting for first stream frame"
                    ));
                }
            }

            latest
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .ok_or_else(|| XCapError::capture_failed("no frame after wait"))
        } else {
            Err(XCapError::capture_failed(
                "stream disappeared after creation",
            ))
        }
    }

    fn ensure_stream(
        monitor_id: u32,
        width: u32,
        height: u32,
        excluded_window_ids: &[u32],
    ) -> XCapResult<()> {
        let mut streams = MANAGER
            .streams
            .lock()
            .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;

        // Remove old stream if params changed
        if let Some(existing) = streams.get(&monitor_id) {
            if existing.width == width
                && existing.height == height
                && existing.excluded_window_ids == excluded_window_ids
            {
                return Ok(()); // already have it
            }
            info!(
                "recreating stream for display {} (params change)",
                monitor_id
            );
        }
        streams.remove(&monitor_id);

        // Enumerate displays to find the target
        let content = get_shareable_content()?;
        let displays = content.displays();
        let sc_display = displays
            .iter()
            .find(|d| d.display_id().0 == monitor_id)
            .ok_or_else(|| XCapError::monitor_not_found(monitor_id))?;

        let ms = MonitorStream::new(sc_display, &content, width, height, 2, excluded_window_ids)?;
        streams.insert(monitor_id, ms);

        Ok(())
    }

    /// Stop and remove a specific monitor stream.
    pub fn invalidate(monitor_id: u32) {
        if let Ok(mut streams) = MANAGER.streams.lock() {
            if streams.remove(&monitor_id).is_some() {
                info!("invalidated persistent stream for display {}", monitor_id);
            }
        }
    }

    /// Stop all streams (for clean shutdown or DRM pause).
    pub fn stop_all() {
        if let Ok(mut streams) = MANAGER.streams.lock() {
            let count = streams.len();
            streams.clear();
            if count > 0 {
                info!("stopped {} persistent stream(s)", count);
            }
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────

/// Capture a monitor frame using a persistent SCStream.
///
/// On first call, creates the stream and waits for the first frame.
/// Subsequent calls return the latest buffered frame immediately.
///
/// `excluded_window_ids` — SCK window IDs to exclude from the capture.
/// If the exclusion list differs from the running stream, the stream is
/// recreated with the new filter.
pub async fn capture_monitor_persistent(
    monitor_id: u32,
    width: u32,
    height: u32,
    excluded_window_ids: &[u32],
) -> XCapResult<RgbaImage> {
    StreamManager::capture(monitor_id, width, height, excluded_window_ids).await
}

/// Stop the persistent stream for a monitor (e.g. for DRM pause).
/// The stream will be recreated on the next capture call.
pub fn invalidate_monitor_stream(monitor_id: u32) {
    StreamManager::invalidate(monitor_id);
}

/// Stop all persistent streams (for shutdown or DRM pause).
pub fn stop_all_streams() {
    StreamManager::stop_all();
}
