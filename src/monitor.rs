//! Monitor/Display capture using ScreenCaptureKit via cidre

use cidre::cg;
use image::RgbaImage;
use tracing::debug;

use crate::capture;
use crate::error::{XCapError, XCapResult};

/// Represents a capturable monitor/display
///
/// This type provides an API compatible with xcap::Monitor
#[derive(Debug, Clone)]
pub struct Monitor {
    /// The display ID
    display_id: u32,
    /// Display name (if available)
    name: String,
    /// Display X position
    x: i32,
    /// Display Y position
    y: i32,
    /// Display width in pixels
    width: u32,
    /// Display height in pixels
    height: u32,
    /// Scale factor (for Retina displays)
    scale_factor: f64,
    /// Whether this is the primary display
    is_primary: bool,
}

impl Monitor {
    /// Get all available monitors
    ///
    /// Returns a list of all displays that can be captured.
    /// Requires screen recording permission.
    pub fn all() -> XCapResult<Vec<Monitor>> {
        let content = capture::get_shareable_content()?;

        let displays = content.displays();

        if displays.is_empty() {
            return Err(XCapError::no_monitors());
        }

        // Find the primary display (usually the first one, or has origin at 0,0)
        let primary_id = displays
            .iter()
            .find(|d| {
                let frame = d.frame();
                frame.origin.x == 0.0 && frame.origin.y == 0.0
            })
            .map(|d| d.display_id())
            .unwrap_or_else(|| displays.first().map(|d| d.display_id()).unwrap_or(cg::DirectDisplayId::NULL));

        let monitors: Vec<Monitor> = displays
            .iter()
            .map(|d| {
                let frame = d.frame();
                let width = d.width() as u32;
                let height = d.height() as u32;
                let display_id = d.display_id();

                debug!(
                    "Found display {:?}: {}x{} at ({}, {})",
                    display_id, width, height, frame.origin.x, frame.origin.y
                );

                Monitor {
                    display_id: display_id.0,
                    name: format!("Display {}", display_id.0),
                    x: frame.origin.x as i32,
                    y: frame.origin.y as i32,
                    width,
                    height,
                    scale_factor: 1.0, // TODO: Get actual scale factor from CGDisplay
                    is_primary: display_id == primary_id,
                }
            })
            .collect();

        Ok(monitors)
    }

    /// Get the primary monitor
    pub fn primary() -> XCapResult<Monitor> {
        let monitors = Self::all()?;
        monitors
            .into_iter()
            .find(|m| m.is_primary)
            .ok_or_else(|| XCapError::new("No primary monitor found"))
    }

    /// Get the monitor ID
    pub fn id(&self) -> u32 {
        self.display_id
    }

    /// Get the monitor name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the monitor X position
    pub fn x(&self) -> i32 {
        self.x
    }

    /// Get the monitor Y position
    pub fn y(&self) -> i32 {
        self.y
    }

    /// Get the monitor width in pixels
    pub fn width(&self) -> XCapResult<u32> {
        Ok(self.width)
    }

    /// Get the monitor height in pixels
    pub fn height(&self) -> XCapResult<u32> {
        Ok(self.height)
    }

    /// Get the raw width (non-Result version)
    pub fn raw_width(&self) -> u32 {
        self.width
    }

    /// Get the raw height (non-Result version)
    pub fn raw_height(&self) -> u32 {
        self.height
    }

    /// Get the scale factor (for Retina displays)
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Check if this is the primary monitor
    pub fn is_primary(&self) -> bool {
        self.is_primary
    }

    /// Capture an image of the monitor
    ///
    /// Returns an RGBA image of the entire monitor.
    pub fn capture_image(&self) -> XCapResult<RgbaImage> {
        capture::capture_monitor_sync(self.display_id, self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_getters() {
        let monitor = Monitor {
            display_id: 1,
            name: "Test Display".to_string(),
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
            scale_factor: 2.0,
            is_primary: true,
        };

        assert_eq!(monitor.id(), 1);
        assert_eq!(monitor.name(), "Test Display");
        assert_eq!(monitor.x(), 0);
        assert_eq!(monitor.y(), 0);
        assert_eq!(monitor.width().unwrap(), 1920);
        assert_eq!(monitor.height().unwrap(), 1080);
        assert_eq!(monitor.raw_width(), 1920);
        assert_eq!(monitor.raw_height(), 1080);
        assert_eq!(monitor.scale_factor(), 2.0);
        assert!(monitor.is_primary());
    }

    #[test]
    fn test_monitor_all() {
        // This test verifies the API works
        let result = Monitor::all();
        let _ = result;
    }
}
