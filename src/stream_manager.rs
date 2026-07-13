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
    api, arc, cm, cv, define_obj_type, dispatch, ns, objc, sc,
    sc::stream::{Output, OutputImpl},
};
use image::RgbaImage;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
    /// When set, the receiver runs in *push* mode: every delivered frame is
    /// forwarded to this channel (newest dropped on backpressure) instead of
    /// latching `latest_frame`. Used by the high-fps HD capture stream so a
    /// recorder can consume every frame. `None` = the default latch mode used
    /// by the persistent screenshot stream.
    frame_tx: Option<tokio::sync::mpsc::Sender<RgbaImage>>,
    /// Monotonic count of frames the OS callback has latched (latch mode only).
    /// Bumped once per delivered frame, so a consumer can tell whether the
    /// stream is still being fed: if this stops advancing while the stream is
    /// alive, the OS callback has wedged and `latest_frame` is stale. A static
    /// screen still advances it — ScreenCaptureKit delivers identical frames at
    /// the frame interval — which is what makes "seq stalled" mean "frozen
    /// stream", not "idle screen".
    frame_seq: Arc<AtomicU64>,
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

        let inner = self.inner_mut();
        // Push mode (HD): forward every frame to the channel and return. Drop
        // the newest frame on backpressure (`try_send`) so a slow encoder never
        // stalls this OS callback queue — a dropped frame is just a tiny replay
        // gap, never a capture hang.
        if let Some(tx) = inner.frame_tx.as_ref() {
            let _ = tx.try_send(image);
            return;
        }
        // Latch mode (default): keep only the latest frame + wake waiters.
        if let Ok(mut guard) = inner.latest_frame.lock() {
            *guard = Some(image);
        }
        // Bump the delivery sequence so consumers can detect a wedged callback
        // (seq stops advancing) vs. a static screen (seq keeps advancing).
        inner.frame_seq.fetch_add(1, Ordering::Release);
        inner.frame_notify.notify_waiters();
    }
}

// A ScreenCaptureKit stop completion is normally prompt, but macOS can fail to
// deliver it after a capture callback wedges. Keep stop asynchronous so a
// missing completion can never block capture recovery or the app's main
// thread. The cap prevents repeated OS failures from retaining an unbounded
// number of streams/completion blocks.
const MAX_PENDING_STREAM_STOPS: usize = 8;

struct StopLimiter {
    pending: AtomicUsize,
    max: usize,
}

impl StopLimiter {
    const fn new(max: usize) -> Self {
        Self {
            pending: AtomicUsize::new(0),
            max,
        }
    }

    fn try_acquire(&self) -> Option<PendingStopSlot<'_>> {
        let mut current = self.pending.load(Ordering::Acquire);
        loop {
            if current >= self.max {
                return None;
            }
            match self.pending.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(PendingStopSlot { limiter: self }),
                Err(observed) => current = observed,
            }
        }
    }

    #[cfg(test)]
    fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }
}

struct PendingStopSlot<'a> {
    limiter: &'a StopLimiter,
}

impl Drop for PendingStopSlot<'_> {
    fn drop(&mut self) {
        self.limiter.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

static STREAM_STOP_LIMITER: StopLimiter = StopLimiter::new(MAX_PENDING_STREAM_STOPS);

// ── Per-monitor stream ─────────────────────────────────────────────

struct MonitorStream {
    _stream: arc::R<sc::Stream>,
    _output: arc::R<FrameReceiver>,
    _queue: arc::R<dispatch::Queue>,
    latest_frame: Arc<Mutex<Option<RgbaImage>>>,
    frame_notify: Arc<tokio::sync::Notify>,
    /// Shared with the FrameReceiver callback; see `FrameReceiverInner::frame_seq`.
    frame_seq: Arc<AtomicU64>,
    width: u32,
    height: u32,
    /// Sorted SCK window IDs currently excluded from this stream's ContentFilter.
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
        frame_tx: Option<tokio::sync::mpsc::Sender<RgbaImage>>,
    ) -> XCapResult<Self> {
        let latest_frame = Arc::new(Mutex::new(None));
        let frame_notify = Arc::new(tokio::sync::Notify::new());
        let frame_seq = Arc::new(AtomicU64::new(0));

        let inner = FrameReceiverInner {
            latest_frame: latest_frame.clone(),
            frame_notify: frame_notify.clone(),
            frame_seq: frame_seq.clone(),
            frame_tx,
        };
        let output = FrameReceiver::with(inner);

        let queue = dispatch::Queue::serial_with_ar_pool();

        // Configure stream
        let mut cfg = sc::StreamCfg::new();
        cfg.set_width(width as usize);
        cfg.set_height(height as usize);
        cfg.set_pixel_format(cv::PixelFormat::_32_BGRA);
        cfg.set_shows_cursor(false);
        if api::macos_available("15.0") {
            cfg.set_show_mouse_clicks(false);
        }
        // scales_to_fit(true) so callers can request a downscaled capture
        // (target_w/target_h < native) and have the GPU do the resize before
        // the framebuffer hits replayd. Major WindowServer/replayd cost saver
        // on HiDPI displays and for OCR-quality (not pixel-perfect) consumers.
        // No-op when the requested dims equal native.
        cfg.set_scales_to_fit(true);
        cfg.set_minimum_frame_interval(cm::Time::new(1, fps));

        let filter = build_exclusion_filter(sc_display, content, excluded_window_ids);

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

        let mut sorted_ids = excluded_window_ids.to_vec();
        sorted_ids.sort_unstable();

        Ok(Self {
            _stream: stream,
            _output: output,
            _queue: queue,
            latest_frame,
            frame_notify,
            frame_seq,
            width,
            height,
            excluded_window_ids: sorted_ids,
        })
    }

    /// Update the exclusion filter on a running stream without recreating it.
    fn update_exclusions(
        &mut self,
        new_ids: &[u32],
        sc_display: &sc::Display,
        content: &sc::ShareableContent,
    ) -> XCapResult<()> {
        let filter = build_exclusion_filter(sc_display, content, new_ids);

        let stream = self._stream.retained();
        crate::capture::run_in_thread(move || {
            crate::capture::block_on(async { stream.update_content_filter(&filter).await })
        })
        .map_err(|e| XCapError::capture_failed(format!("failed to update content filter: {}", e)))?
        .map_err(|e| XCapError::capture_failed(format!("update content filter error: {:?}", e)))?;

        let mut sorted = new_ids.to_vec();
        sorted.sort_unstable();
        self.excluded_window_ids = sorted;

        debug!(
            "updated exclusion filter in-place ({} excluded)",
            new_ids.len()
        );

        Ok(())
    }

    fn latest_frame(&self) -> Option<RgbaImage> {
        self.latest_frame.lock().ok()?.clone()
    }

    /// Monotonic count of frames the OS callback has latched. See
    /// `FrameReceiverInner::frame_seq`.
    fn frame_seq(&self) -> u64 {
        self.frame_seq.load(Ordering::Acquire)
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
        let Some(stop_slot) = STREAM_STOP_LIMITER.try_acquire() else {
            // Dropping the final SCStream handle is the only bounded fallback
            // once macOS has already stranded too many stop completions.
            warn!(
                "too many pending SCK stream stops; releasing stream without waiting for completion"
            );
            return;
        };

        debug!("requesting asynchronous persistent SCK stream stop");
        let stream = self._stream.retained();
        let mut keep_stream_alive = Some(stream.retained());
        let mut keep_output_alive = Some(self._output.retained());
        let mut keep_queue_alive = Some(self._queue.retained());
        let mut stop_slot = Some(stop_slot);
        stream.stop_with_ch(move |error| {
            if let Some(error) = error {
                warn!("persistent SCK stream stop failed: {:?}", error);
            } else {
                debug!("persistent SCK stream stopped");
            }

            // The callback can legally be invoked only once, but use Options
            // so an unexpected duplicate callback remains harmless.
            let _ = keep_queue_alive.take();
            let _ = keep_output_alive.take();
            let _ = keep_stream_alive.take();
            let _ = stop_slot.take();
        });
    }
}

/// Build a ContentFilter that excludes the given window IDs.
fn build_exclusion_filter(
    sc_display: &sc::Display,
    content: &sc::ShareableContent,
    excluded_window_ids: &[u32],
) -> arc::R<sc::ContentFilter> {
    if excluded_window_ids.is_empty() {
        let empty = ns::Array::new();
        return sc::ContentFilter::with_display_excluding_windows(sc_display, &empty);
    }

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
            "exclusion filter: {} window(s) for display {}",
            to_exclude.len(),
            sc_display.display_id().0
        );
        let arr = ns::Array::from_slice(&to_exclude);
        sc::ContentFilter::with_display_excluding_windows(sc_display, &arr)
    }
}

// ── Stream manager (singleton) ─────────────────────────────────────

struct StreamManager {
    streams: Mutex<HashMap<u32, MonitorStream>>,
}

static MANAGER: Lazy<StreamManager> = Lazy::new(|| StreamManager {
    streams: Mutex::new(HashMap::new()),
});

fn remove_entry<K, V>(entries: &Mutex<HashMap<K, V>>, key: &K) -> Option<V>
where
    K: Eq + Hash,
{
    entries.lock().ok()?.remove(key)
}

fn take_entries<K, V>(entries: &Mutex<HashMap<K, V>>) -> Option<HashMap<K, V>> {
    let mut entries = entries.lock().ok()?;
    Some(std::mem::take(&mut *entries))
}

impl StreamManager {
    /// Get the latest frame for a monitor, creating the stream if needed.
    ///
    /// On first call for a monitor, creates a persistent SCStream and waits
    /// up to 3s for the first frame. Subsequent calls return the latest
    /// buffered frame immediately.
    ///
    /// If `excluded_window_ids` changes, the content filter is updated
    /// in-place on the running stream (no stop/start). The stream is only
    /// fully recreated when resolution changes.
    pub async fn capture(
        monitor_id: u32,
        width: u32,
        height: u32,
        excluded_window_ids: &[u32],
    ) -> XCapResult<RgbaImage> {
        let mut sorted_input = excluded_window_ids.to_vec();
        sorted_input.sort_unstable();

        // Fast path: stream exists, matches params, and has a frame
        {
            let mut streams = MANAGER
                .streams
                .lock()
                .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
            if let Some(ms) = streams.get_mut(&monitor_id) {
                if ms.width != width || ms.height != height {
                    // Resolution changed — must fully recreate
                    debug!(
                        "resolution changed for display {}, recreating stream",
                        monitor_id
                    );
                } else if ms.excluded_window_ids == sorted_input {
                    // Exact match — return latest frame
                    if let Some(frame) = ms.latest_frame() {
                        return Ok(frame);
                    }
                    // Stream exists but no frame yet — wait below
                    let notify = ms.frame_notify.clone();
                    let latest = ms.latest_frame.clone();
                    drop(streams);
                    return Self::wait_for_frame(notify, latest).await;
                } else {
                    // Exclusions changed — update filter in-place
                    let content = get_shareable_content()?;
                    let displays = content.displays();
                    if let Some(sc_display) =
                        displays.iter().find(|d| d.display_id().0 == monitor_id)
                    {
                        match ms.update_exclusions(excluded_window_ids, sc_display, &content) {
                            Ok(()) => {
                                if let Some(frame) = ms.latest_frame() {
                                    return Ok(frame);
                                }
                                let notify = ms.frame_notify.clone();
                                let latest = ms.latest_frame.clone();
                                drop(streams);
                                return Self::wait_for_frame(notify, latest).await;
                            }
                            Err(e) => {
                                warn!(
                                    "failed to update filter in-place for display {}: {}, recreating",
                                    monitor_id, e
                                );
                                // Fall through to full recreation
                            }
                        }
                    }
                }
            }
        }

        // Slow path: create or recreate stream
        Self::create_stream(monitor_id, width, height, excluded_window_ids)?;

        // Wait for first frame
        let streams = MANAGER
            .streams
            .lock()
            .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
        if let Some(ms) = streams.get(&monitor_id) {
            let notify = ms.frame_notify.clone();
            let latest = ms.latest_frame.clone();
            drop(streams);
            Self::wait_for_frame(notify, latest).await
        } else {
            Err(XCapError::capture_failed(
                "stream disappeared after creation",
            ))
        }
    }

    async fn wait_for_frame(
        notify: Arc<tokio::sync::Notify>,
        latest: Arc<Mutex<Option<RgbaImage>>>,
    ) -> XCapResult<RgbaImage> {
        // Check if frame is already available
        if let Some(frame) = latest.lock().ok().and_then(|g| g.clone()) {
            return Ok(frame);
        }

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
    }

    fn create_stream(
        monitor_id: u32,
        width: u32,
        height: u32,
        excluded_window_ids: &[u32],
    ) -> XCapResult<()> {
        let mut streams = MANAGER
            .streams
            .lock()
            .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;

        // Check again under lock
        if let Some(existing) = streams.get(&monitor_id) {
            let mut sorted = excluded_window_ids.to_vec();
            sorted.sort_unstable();
            if existing.width == width
                && existing.height == height
                && existing.excluded_window_ids == sorted
            {
                return Ok(());
            }
            info!(
                "recreating stream for display {} (resolution change)",
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

        let ms = MonitorStream::new(
            sc_display,
            &content,
            width,
            height,
            2,
            excluded_window_ids,
            None, // latch mode — the persistent screenshot stream
        )?;
        streams.insert(monitor_id, ms);

        Ok(())
    }

    /// Stop and remove a specific monitor stream.
    pub fn invalidate(monitor_id: u32) {
        // Keep the removed value alive until after the mutex guard is gone.
        // Its destructor initiates ScreenCaptureKit teardown and must never run
        // while readers such as the tray preview are excluded from the map.
        let removed = remove_entry(&MANAGER.streams, &monitor_id);
        let had_stream = removed.is_some();
        drop(removed);
        if had_stream {
            info!("invalidated persistent stream for display {}", monitor_id);
        }
    }

    /// Stop all streams (for clean shutdown or DRM pause).
    pub fn stop_all() {
        // Move the map out atomically, then run every destructor after the
        // global lock has been released.
        let streams = take_entries(&MANAGER.streams).unwrap_or_default();
        let count = streams.len();
        drop(streams);
        if count > 0 {
            info!("stopped {} persistent stream(s)", count);
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
/// If the exclusion list changes, the content filter is updated in-place
/// on the running stream (no teardown). The stream is only recreated
/// when the resolution changes.
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

/// Current frame-delivery sequence for a monitor's persistent stream, if one
/// is cached. Monotonic; bumped once per OS-latched frame. `None` when no
/// stream exists for this monitor yet.
///
/// Compare it across captures: if it does not advance while captures keep
/// happening, the stream's OS callback has wedged and
/// `capture_monitor_persistent` is returning a stale frame — invalidate and
/// recreate the stream. A static screen keeps advancing it (SCK delivers
/// identical frames at the frame interval), so a stalled sequence means a dead
/// stream, not an idle one.
pub fn monitor_frame_seq(monitor_id: u32) -> Option<u64> {
    MANAGER
        .streams
        .lock()
        .ok()?
        .get(&monitor_id)
        .map(|ms| ms.frame_seq())
}

/// Return the latest latched RGBA frame from an already-running persistent
/// stream, without waiting or creating a stream. `None` when no stream exists
/// yet or no frame has been delivered.
pub fn peek_latest_frame(monitor_id: u32) -> Option<RgbaImage> {
    MANAGER
        .streams
        .try_lock()
        .ok()?
        .get(&monitor_id)
        .and_then(|ms| ms.latest_frame())
}

// ── High-FPS HD capture (push mode) ────────────────────────────────

/// Bounded frame channel depth for the HD capture stream. Small on purpose:
/// the recorder should consume in real time, so we'd rather drop the newest
/// frame than let memory grow if it briefly stalls. 8 frames ≈ 0.8s at 10fps.
const HD_CHANNEL_CAPACITY: usize = 8;

/// Maximum HD capture rate we'll request from ScreenCaptureKit.
const HD_MAX_FPS: u32 = 60;

/// A standalone high-frame-rate capture stream.
///
/// Unlike the persistent screenshot stream (2 fps, latest-frame-only), this
/// delivers EVERY frame to a channel so a recorder can encode continuous
/// video. It is independent of the `StreamManager` singleton — a second
/// `SCStream` coexists with the screenshot stream on the same display (SCK
/// supports multiple concurrent streams per display), so HD recording never
/// disturbs the OCR/screenshot path. Drop the handle to stop capture.
pub struct HdCaptureStream {
    _inner: MonitorStream,
    monitor_id: u32,
    fps: u32,
}

impl HdCaptureStream {
    /// The display this stream is capturing.
    pub fn monitor_id(&self) -> u32 {
        self.monitor_id
    }
    /// The frame rate the stream was started at.
    pub fn fps(&self) -> u32 {
        self.fps
    }
}

/// Start a dedicated high-fps capture stream for `monitor_id` at `fps`, with
/// frames GPU-downscaled to `width`x`height`.
///
/// Returns the stream handle (drop to stop) and a receiver of RGBA frames.
/// Under backpressure the newest frame is dropped (`HD_CHANNEL_CAPACITY`), so a
/// slow consumer can never stall the OS callback. `excluded_window_ids` are
/// excluded at the OS level — ignored/private windows never reach the recorder.
///
/// This blocks briefly while ScreenCaptureKit starts the stream; call it from a
/// blocking context (e.g. `spawn_blocking`).
pub fn start_hd_capture(
    monitor_id: u32,
    width: u32,
    height: u32,
    fps: u32,
    excluded_window_ids: &[u32],
) -> XCapResult<(HdCaptureStream, tokio::sync::mpsc::Receiver<RgbaImage>)> {
    let clamped_fps = fps.clamp(1, HD_MAX_FPS);
    let (tx, rx) = tokio::sync::mpsc::channel(HD_CHANNEL_CAPACITY);

    let content = get_shareable_content()?;
    let displays = content.displays();
    let sc_display = displays
        .iter()
        .find(|d| d.display_id().0 == monitor_id)
        .ok_or_else(|| XCapError::monitor_not_found(monitor_id))?;

    let stream = MonitorStream::new(
        sc_display,
        &content,
        width,
        height,
        clamped_fps as i32,
        excluded_window_ids,
        Some(tx), // push mode — every frame to the channel
    )?;

    info!(
        "HD capture stream started for display {} ({}x{} @ {}fps)",
        monitor_id, width, height, clamped_fps
    );

    Ok((
        HdCaptureStream {
            _inner: stream,
            monitor_id,
            fps: clamped_fps,
        },
        rx,
    ))
}

#[cfg(test)]
mod stream_manager_regression_tests {
    use super::*;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::Barrier;
    use std::thread;
    use std::time::{Duration, Instant};

    struct BlockingDrop {
        entered: Sender<()>,
        release: Receiver<()>,
    }

    impl Drop for BlockingDrop {
        fn drop(&mut self) {
            let _ = self.entered.send(());
            let _ = self.release.recv_timeout(Duration::from_secs(2));
        }
    }

    #[test]
    fn peek_latest_frame_does_not_wait_for_manager_lock() {
        let _manager_guard = MANAGER.streams.lock().expect("manager lock");
        let started = Instant::now();

        assert!(peek_latest_frame(u32::MAX).is_none());
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "peek_latest_frame must be non-blocking while capture teardown owns the manager lock"
        );
    }

    #[test]
    fn removed_entry_is_dropped_after_its_map_lock_is_released() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let entries = Arc::new(Mutex::new(HashMap::from([(
            7_u32,
            BlockingDrop {
                entered: entered_tx,
                release: release_rx,
            },
        )])));
        let worker_entries = Arc::clone(&entries);

        let worker = thread::spawn(move || {
            let removed = remove_entry(&worker_entries, &7);
            drop(removed);
        });

        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("removed value should begin dropping");
        assert!(
            entries.try_lock().is_ok(),
            "map lock must be available while a removed value's destructor is blocked"
        );
        release_tx.send(()).expect("release blocking drop");
        worker.join().expect("drop worker");
    }

    #[test]
    fn drained_entries_are_dropped_after_their_map_lock_is_released() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let entries = Arc::new(Mutex::new(HashMap::from([(
            11_u32,
            BlockingDrop {
                entered: entered_tx,
                release: release_rx,
            },
        )])));
        let worker_entries = Arc::clone(&entries);

        let worker = thread::spawn(move || {
            let drained = take_entries(&worker_entries).expect("drain map");
            drop(drained);
        });

        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("drained value should begin dropping");
        assert!(
            entries.try_lock().is_ok(),
            "map lock must be available while drained values are being destroyed"
        );
        release_tx.send(()).expect("release blocking drop");
        worker.join().expect("drop worker");
    }

    #[test]
    fn pending_stop_limiter_caps_and_releases_slots() {
        let limiter = StopLimiter::new(2);
        let first = limiter.try_acquire().expect("first stop slot");
        let second = limiter.try_acquire().expect("second stop slot");

        assert_eq!(limiter.pending(), 2);
        assert!(limiter.try_acquire().is_none());

        drop(first);
        assert_eq!(limiter.pending(), 1);
        let replacement = limiter.try_acquire().expect("released slot is reusable");
        assert_eq!(limiter.pending(), 2);

        drop(second);
        drop(replacement);
        assert_eq!(limiter.pending(), 0);
    }

    #[test]
    fn pending_stop_limiter_is_race_safe() {
        const THREADS: usize = 32;
        const LIMIT: usize = 4;

        let limiter = Arc::new(StopLimiter::new(LIMIT));
        let barrier = Arc::new(Barrier::new(THREADS));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let rejected = Arc::new(AtomicUsize::new(0));

        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let limiter = Arc::clone(&limiter);
                let barrier = Arc::clone(&barrier);
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                let rejected = Arc::clone(&rejected);
                thread::spawn(move || {
                    barrier.wait();
                    let Some(slot) = limiter.try_acquire() else {
                        rejected.fetch_add(1, Ordering::AcqRel);
                        return;
                    };
                    let current = active.fetch_add(1, Ordering::AcqRel) + 1;
                    max_active.fetch_max(current, Ordering::AcqRel);
                    thread::sleep(Duration::from_millis(20));
                    active.fetch_sub(1, Ordering::AcqRel);
                    drop(slot);
                })
            })
            .collect();

        for worker in workers {
            worker.join().expect("stop limiter worker");
        }

        assert!(max_active.load(Ordering::Acquire) <= LIMIT);
        assert!(rejected.load(Ordering::Acquire) > 0);
        assert_eq!(limiter.pending(), 0);
    }
}
