//! Monitor/Display capture using ScreenCaptureKit via cidre

use core_graphics::display::CGDisplay;
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
    /// Display width in physical pixels (actual framebuffer resolution)
    width: u32,
    /// Display height in physical pixels (actual framebuffer resolution)
    height: u32,
    /// Logical width (what macOS reports for UI layout)
    logical_width: u32,
    /// Logical height (what macOS reports for UI layout)
    logical_height: u32,
    /// Scale factor (for Retina displays) - ratio of physical to logical pixels
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
        // Note: display_id() returns DirectDisplayId, use .0 to get the u32 value
        let primary_id = displays
            .iter()
            .find(|d| {
                let frame = d.frame();
                frame.origin.x == 0.0 && frame.origin.y == 0.0
            })
            .map(|d| d.display_id().0)
            .unwrap_or_else(|| displays.first().map(|d| d.display_id().0).unwrap_or(0));

        let monitors: Vec<Monitor> = displays
            .iter()
            .map(|d| {
                let frame = d.frame();
                let display_id = d.display_id().0; // Extract u32 from DirectDisplayId
                
                // Get logical dimensions from ScreenCaptureKit (these are scaled for UI)
                let logical_width = d.width() as u32;
                let logical_height = d.height() as u32;
                
                // Get actual physical pixel dimensions from Core Graphics
                // CGDisplayPixelsWide/High return the actual framebuffer resolution
                let cg_display = CGDisplay::new(display_id);
                let physical_width = cg_display.pixels_wide() as u32;
                let physical_height = cg_display.pixels_high() as u32;
                
                // Calculate scale factor from physical vs logical dimensions
                // For Retina displays, this will be 2.0; for standard displays, 1.0
                // For ultrawide or unusual monitors, calculate based on actual ratio
                let scale_factor = if logical_width > 0 && logical_height > 0 {
                    let width_scale = physical_width as f64 / logical_width as f64;
                    let height_scale = physical_height as f64 / logical_height as f64;
                    // Use the average to handle any rounding differences
                    (width_scale + height_scale) / 2.0
                } else {
                    1.0
                };

                debug!(
                    "Found display {}: physical={}x{}, logical={}x{}, scale={:.2}, at ({}, {})",
                    display_id, physical_width, physical_height, 
                    logical_width, logical_height, scale_factor,
                    frame.origin.x, frame.origin.y
                );

                Monitor {
                    display_id,
                    name: format!("Display {}", display_id),
                    x: frame.origin.x as i32,
                    y: frame.origin.y as i32,
                    // Use physical pixels for capture (actual screen resolution)
                    width: physical_width,
                    height: physical_height,
                    logical_width,
                    logical_height,
                    scale_factor,
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

    /// Get the raw width (non-Result version) - returns physical pixels
    pub fn raw_width(&self) -> u32 {
        self.width
    }

    /// Get the raw height (non-Result version) - returns physical pixels
    pub fn raw_height(&self) -> u32 {
        self.height
    }

    /// Get the logical width (what macOS reports for UI layout)
    pub fn logical_width(&self) -> u32 {
        self.logical_width
    }

    /// Get the logical height (what macOS reports for UI layout)
    pub fn logical_height(&self) -> u32 {
        self.logical_height
    }

    /// Get the scale factor (for Retina displays)
    /// This is the ratio of physical pixels to logical pixels
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
            width: 3840,      // Physical pixels (4K)
            height: 2160,
            logical_width: 1920,  // Logical pixels (what UI sees)
            logical_height: 1080,
            scale_factor: 2.0,
            is_primary: true,
        };

        assert_eq!(monitor.id(), 1);
        assert_eq!(monitor.name(), "Test Display");
        assert_eq!(monitor.x(), 0);
        assert_eq!(monitor.y(), 0);
        assert_eq!(monitor.width().unwrap(), 3840);
        assert_eq!(monitor.height().unwrap(), 2160);
        assert_eq!(monitor.raw_width(), 3840);
        assert_eq!(monitor.raw_height(), 2160);
        assert_eq!(monitor.logical_width(), 1920);
        assert_eq!(monitor.logical_height(), 1080);
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
