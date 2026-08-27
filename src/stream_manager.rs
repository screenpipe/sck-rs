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

use crate::capture::{await_sck_callback, get_shareable_content, safe_image_buf_to_rgba};
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
    /// Monotonic count of screen callbacks observed (latch mode only). This is
    /// bumped before looking for an image buffer because ScreenCaptureKit sends
    /// idle updates without an IOSurface when the display has not changed. A
    /// consumer can therefore distinguish a live idle stream from a callback
    /// that has actually wedged.
    frame_seq: Arc<AtomicU64>,
}

define_obj_type!(
    FrameReceiver + OutputImpl,
    FrameReceiverInner,
    FRAME_RECEIVER_CLS
);

impl Output for FrameReceiver {}

fn push_channel_accepts_frame(tx: &tokio::sync::mpsc::Sender<RgbaImage>) -> bool {
    tx.capacity() > 0
}

fn mark_latch_stream_callback(
    frame_tx: Option<&tokio::sync::mpsc::Sender<RgbaImage>>,
    frame_seq: &AtomicU64,
) {
    if frame_tx.is_none() {
        frame_seq.fetch_add(1, Ordering::Release);
    }
}

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

        // HD push mode is newest-frame-only. Check the bounded handoff before
        // touching the pixel buffer: BGRA -> RGBA copies the entire display
        // and is the dominant callback cost at meeting capture rates. When the
        // encoder already has one frame waiting, converting another frame only
        // to have try_send reject it wastes a full-frame copy.
        if self
            .inner_mut()
            .frame_tx
            .as_ref()
            .is_some_and(|tx| !push_channel_accepts_frame(tx))
        {
            return;
        }

        // An idle ScreenCaptureKit sample has no image buffer because there is
        // no new IOSurface. It still proves the callback is alive, so advance
        // liveness before the image-buffer early return. HD push streams do not
        // expose this sequence and remain governed by channel backpressure.
        {
            let inner = self.inner_mut();
            mark_latch_stream_callback(inner.frame_tx.as_ref(), &inner.frame_seq);
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

/// Monotonic id source for `MonitorStream::generation`.
static STREAM_GENERATION: AtomicU64 = AtomicU64::new(0);

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
    /// Identity token: exclusion updates are applied outside the map lock, so
    /// the updater re-checks under the lock that the entry it fetched content
    /// for is still the same stream before committing bookkeeping.
    generation: u64,
    /// Single-flight guard for filter updates. Updates run outside the map
    /// lock, so without this two concurrent captures could interleave their
    /// apply and commit phases (apply A, apply B, commit B, commit A) and
    /// leave `excluded_window_ids` describing a filter the OS is not running.
    filter_update_busy: Arc<std::sync::atomic::AtomicBool>,
}

/// Clears a stream's filter-update busy flag on every exit path (including
/// panics and early `?` returns) so a failed update can never wedge future
/// updates behind a stuck flag.
struct FilterUpdateGuard(Arc<std::sync::atomic::AtomicBool>);

impl Drop for FilterUpdateGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
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

        // Start the stream. Bounded: SCStream.startCapture's completion
        // handler can silently never fire on a wedged SCK daemon, and an
        // unbounded join here would freeze the caller (and, transitively,
        // every capture path waiting on the stream map).
        let stream_clone = stream.retained();
        crate::capture::run_bounded("stream-start", Duration::from_secs(10), move || {
            crate::capture::block_on(async {
                await_sck_callback("stream-start", Duration::from_secs(9), stream_clone.start())
                    .await
            })
        })
        .map_err(|e| XCapError::capture_failed(format!("failed to start stream: {}", e)))?
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
            generation: STREAM_GENERATION.fetch_add(1, Ordering::Relaxed),
            filter_update_busy: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Push a new ContentFilter to a running stream, bounded.
    ///
    /// Standalone (no `&self`) so `StreamManager::capture` can run it after
    /// releasing the stream-map lock — `updateContentFilter`'s completion can
    /// wedge exactly like the other SCK callbacks, and holding the map lock
    /// across it froze every capture path in the process.
    fn apply_content_filter(
        stream: arc::R<sc::Stream>,
        filter: arc::R<sc::ContentFilter>,
    ) -> XCapResult<()> {
        crate::capture::run_bounded("filter-update", Duration::from_secs(5), move || {
            crate::capture::block_on(async {
                await_sck_callback(
                    "filter-update",
                    Duration::from_secs(4),
                    stream.update_content_filter(&filter),
                )
                .await
            })
        })
        .map_err(|e| XCapError::capture_failed(format!("failed to update content filter: {}", e)))?
        .map_err(|e| XCapError::capture_failed(format!("failed to update content filter: {}", e)))?
        .map_err(|e| XCapError::capture_failed(format!("update content filter error: {:?}", e)))
    }

    fn latest_frame(&self) -> Option<RgbaImage> {
        self.latest_frame.lock().ok()?.clone()
    }

    /// Monotonic count of screen callbacks the OS has delivered. See
    /// `FrameReceiverInner::frame_seq`.
    fn frame_seq(&self) -> u64 {
        self.frame_seq.load(Ordering::Acquire)
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

        // Phase 1 — classify under the lock, cloning only cheap handles.
        //
        // No ScreenCaptureKit call may run while the map lock is held: SCK
        // completion handlers can silently never fire, and a wedged call
        // holding this lock freezes every capture path in the process
        // (observed in production as the desktop capture loop going silent
        // whenever the window-exclusion set changed on an unresponsive
        // daemon).
        enum Plan {
            /// No usable stream — create or recreate below.
            Create,
            /// Stream matches but has no frame latched yet.
            Wait(Arc<tokio::sync::Notify>, Arc<Mutex<Option<RgbaImage>>>),
            /// Stream matches and a frame is latched.
            Done(RgbaImage),
            /// Exclusion set changed — push a new filter outside the lock.
            UpdateFilter {
                stream: arc::R<sc::Stream>,
                generation: u64,
                fallback_frame: Option<RgbaImage>,
                /// Cleared on every exit via `FilterUpdateGuard`.
                busy: Arc<std::sync::atomic::AtomicBool>,
                /// True when the new set excludes windows the current filter
                /// does not — serving frames without them could leak content
                /// the caller asked to hide, so failures must fail closed.
                exclusions_added: bool,
            },
        }

        let plan = {
            let streams = MANAGER
                .streams
                .lock()
                .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
            match streams.get(&monitor_id) {
                None => Plan::Create,
                Some(ms) if ms.width != width || ms.height != height => {
                    // Resolution changed — must fully recreate
                    debug!(
                        "resolution changed for display {}, recreating stream",
                        monitor_id
                    );
                    Plan::Create
                }
                Some(ms) if ms.excluded_window_ids == sorted_input => match ms.latest_frame() {
                    Some(frame) => Plan::Done(frame),
                    None => Plan::Wait(ms.frame_notify.clone(), ms.latest_frame.clone()),
                },
                Some(ms) => {
                    // Single-flight: only one filter update per stream may be
                    // in flight. A second caller serves the latched frame
                    // (captured under the last COMMITTED filter) instead of
                    // interleaving apply/commit phases with the first.
                    if ms
                        .filter_update_busy
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_err()
                    {
                        match ms.latest_frame() {
                            Some(frame) => Plan::Done(frame),
                            None => Plan::Wait(ms.frame_notify.clone(), ms.latest_frame.clone()),
                        }
                    } else {
                        Plan::UpdateFilter {
                            stream: ms._stream.retained(),
                            generation: ms.generation,
                            fallback_frame: ms.latest_frame(),
                            busy: ms.filter_update_busy.clone(),
                            exclusions_added: sorted_input
                                .iter()
                                .any(|id| !ms.excluded_window_ids.contains(id)),
                        }
                    }
                }
            }
        };

        match plan {
            Plan::Done(frame) => return Ok(frame),
            Plan::Wait(notify, latest) => return Self::wait_for_frame(notify, latest).await,
            Plan::Create => {}
            Plan::UpdateFilter {
                stream,
                generation,
                fallback_frame,
                busy,
                exclusions_added,
            } => {
                // Cleared on every exit path — a failed or panicking update
                // must never wedge future updates behind a stuck busy flag.
                let _busy_guard = FilterUpdateGuard(busy);

                // Phase 2 — all SCK work outside the lock, bounded.
                let update_result = (|| -> XCapResult<()> {
                    let content = get_shareable_content()?;
                    let displays = content.displays();
                    let sc_display = displays
                        .iter()
                        .find(|d| d.display_id().0 == monitor_id)
                        .ok_or_else(|| XCapError::monitor_not_found(monitor_id))?;
                    let filter = build_exclusion_filter(sc_display, &content, excluded_window_ids);
                    MonitorStream::apply_content_filter(stream, filter)
                })();

                match update_result {
                    Ok(()) => {
                        // Phase 3 — commit bookkeeping iff the entry is still
                        // the same stream we pushed the filter to. Guard scope
                        // ends before any await (clippy: await_holding_lock).
                        enum Commit {
                            Frame(RgbaImage),
                            Wait(Arc<tokio::sync::Notify>, Arc<Mutex<Option<RgbaImage>>>),
                            StreamChanged,
                        }
                        let commit = {
                            let mut streams = MANAGER.streams.lock().map_err(|_| {
                                XCapError::capture_failed("stream manager lock poisoned")
                            })?;
                            match streams.get_mut(&monitor_id) {
                                Some(ms) if ms.generation == generation => {
                                    ms.excluded_window_ids = sorted_input.clone();
                                    debug!(
                                        "updated exclusion filter in-place ({} excluded)",
                                        sorted_input.len()
                                    );
                                    match ms.latest_frame() {
                                        Some(frame) => Commit::Frame(frame),
                                        None => Commit::Wait(
                                            ms.frame_notify.clone(),
                                            ms.latest_frame.clone(),
                                        ),
                                    }
                                }
                                _ => Commit::StreamChanged,
                            }
                        };
                        match commit {
                            Commit::Frame(frame) => return Ok(frame),
                            Commit::Wait(notify, latest) => {
                                return Self::wait_for_frame(notify, latest).await
                            }
                            // The stream changed under us — fall through to
                            // the create path, which re-validates params.
                            Commit::StreamChanged => {}
                        }
                    }
                    Err(e) => {
                        // The update FAILED — but "failed" here means "did not
                        // complete in time", not "was not applied": a timed-out
                        // updateContentFilter runs on an abandoned worker and
                        // can still land on the live stream seconds later. The
                        // stream's true filter state is now UNKNOWN, so the
                        // entry must be poisoned: remove it (teardown outside
                        // the lock, same rule as `invalidate`) so no future
                        // capture can take the Plan::Done equality fast path
                        // against a filter the OS may not be running. The next
                        // capture recreates the stream with the correct filter
                        // from scratch (bounded).
                        warn!(
                            "filter update failed for display {} ({}); invalidating stream (live filter state unknown)",
                            monitor_id, e
                        );
                        let removed = remove_entry(&MANAGER.streams, &monitor_id);
                        drop(removed);

                        if !exclusions_added {
                            // Exclusions were only REMOVED: the latched frame
                            // was captured under the stricter previous filter,
                            // so serving it once more never leaks. With newly
                            // ADDED exclusions we fail closed instead and fall
                            // through to recreation with the new filter.
                            if let Some(frame) = fallback_frame {
                                return Ok(frame);
                            }
                        }
                    }
                }
            }
        }

        // Slow path: create or recreate stream
        Self::create_stream(monitor_id, width, height, excluded_window_ids)?;

        // Wait for first frame. Guard scope ends before the await
        // (clippy: await_holding_lock).
        let handles = {
            let streams = MANAGER
                .streams
                .lock()
                .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
            streams
                .get(&monitor_id)
                .map(|ms| (ms.frame_notify.clone(), ms.latest_frame.clone()))
        };
        match handles {
            Some((notify, latest)) => Self::wait_for_frame(notify, latest).await,
            None => Err(XCapError::capture_failed(
                "stream disappeared after creation",
            )),
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
        let mut sorted = excluded_window_ids.to_vec();
        sorted.sort_unstable();

        // Idempotence check under the lock, then remove any stale entry.
        // The removed stream's ScreenCaptureKit teardown runs after the
        // guard is released (same rule as `invalidate`), and no SCK call
        // ever runs while the map lock is held — a wedged completion here
        // used to freeze every capture path in the process.
        let removed = {
            let mut streams = MANAGER
                .streams
                .lock()
                .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
            if let Some(existing) = streams.get(&monitor_id) {
                if existing.width == width
                    && existing.height == height
                    && existing.excluded_window_ids == sorted
                {
                    return Ok(());
                }
                info!("recreating stream for display {}", monitor_id);
            }
            streams.remove(&monitor_id)
        };
        drop(removed);

        // All ScreenCaptureKit work outside the lock, bounded.
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

        // Publish. If a concurrent creator raced us, keep whichever stream
        // already matches the requested params; tear the loser down outside
        // the lock.
        let displaced = {
            let mut streams = MANAGER
                .streams
                .lock()
                .map_err(|_| XCapError::capture_failed("stream manager lock poisoned"))?;
            let equivalent_exists = streams.get(&monitor_id).is_some_and(|existing| {
                existing.width == width
                    && existing.height == height
                    && existing.excluded_window_ids == sorted
            });
            if equivalent_exists {
                Some(ms)
            } else {
                streams.insert(monitor_id, ms)
            }
        };
        drop(displaced);

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
/// is cached. Monotonic; bumped once per screen callback, including idle
/// updates that contain no new image buffer. `None` when no stream exists for
/// this monitor yet.
///
/// Compare it across captures: if it does not advance while captures keep
/// happening, the stream's OS callback has wedged and
/// `capture_monitor_persistent` is returning a stale frame — invalidate and
/// recreate the stream. A static screen keeps advancing it through SCK idle
/// callbacks, so a stalled sequence means a dead callback, not an idle display.
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

/// Bounded frame channel depth for the HD capture stream. One queued frame is
/// intentional: the recorder emits constant-rate video by repeating its last
/// encoded frame, so a backlog cannot improve output quality. Keeping only one
/// handoff also lets the callback skip full-frame BGRA -> RGBA conversion while
/// the encoder is busy.
const HD_CHANNEL_CAPACITY: usize = 1;

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

    #[test]
    fn hd_push_channel_applies_backpressure_before_frame_conversion() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        assert!(push_channel_accepts_frame(&tx));

        tx.try_send(RgbaImage::new(2, 2)).unwrap();
        assert!(
            !push_channel_accepts_frame(&tx),
            "a queued frame must suppress another full-frame conversion"
        );

        let _ = rx.try_recv().unwrap();
        assert!(
            push_channel_accepts_frame(&tx),
            "conversion can resume as soon as the recorder drains the handoff"
        );
    }

    #[test]
    fn idle_latch_callback_advances_liveness_without_an_image() {
        let seq = AtomicU64::new(0);

        mark_latch_stream_callback(None, &seq);
        assert_eq!(seq.load(Ordering::Acquire), 1);

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        mark_latch_stream_callback(Some(&tx), &seq);
        assert_eq!(
            seq.load(Ordering::Acquire),
            1,
            "HD push callbacks must not mutate the screenshot liveness sequence"
        );
    }

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
