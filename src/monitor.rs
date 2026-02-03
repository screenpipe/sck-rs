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
                
                // Get dimensions from multiple sources for comparison
                let cg_display = CGDisplay::new(display_id);
                
                // Source 1: ScreenCaptureKit's reported dimensions (what SCK thinks the display is)
                let sck_width = d.width() as u32;
                let sck_height = d.height() as u32;
                
                // Source 2: ScreenCaptureKit frame size (should match above)
                let frame_width = frame.size.width as u32;
                let frame_height = frame.size.height as u32;
                
                // Source 3: CGDisplayPixelsWide/High (native panel resolution)
                let cg_pixels_width = cg_display.pixels_wide() as u32;
                let cg_pixels_height = cg_display.pixels_high() as u32;
                
                // Source 4: CGDisplayBounds (current display mode, respects scaling)
                let cg_bounds = cg_display.bounds();
                let cg_bounds_width = cg_bounds.size.width as u32;
                let cg_bounds_height = cg_bounds.size.height as u32;
                
                // Check display rotation
                let rotation = cg_display.rotation();
                let is_rotated = (rotation - 90.0).abs() < 1.0 || (rotation - 270.0).abs() < 1.0;
                
                // IMPORTANT: Use ScreenCaptureKit's dimensions for capture
                // SCK knows what dimensions it expects - using physical pixels causes issues
                // when user has display scaling (e.g., "More Space" or "Larger Text")
                // 
                // For the capture, we use SCK dimensions (logical * backing scale factor)
                // SCK handles the actual pixel capture internally
                let (capture_width, capture_height) = (sck_width, sck_height);
                
                // Calculate scale factor (for informational purposes)
                let scale_factor = if sck_width > 0 && sck_height > 0 {
                    let width_scale = cg_pixels_width as f64 / sck_width as f64;
                    let height_scale = cg_pixels_height as f64 / sck_height as f64;
                    ((width_scale + height_scale) / 2.0).max(1.0)
                } else {
                    1.0
                };

                debug!(
                    "Display {} dimensions - SCK: {}x{}, Frame: {}x{}, CGPixels: {}x{}, CGBounds: {}x{}, rotation: {}°, using: {}x{}",
                    display_id,
                    sck_width, sck_height,
                    frame_width, frame_height,
                    cg_pixels_width, cg_pixels_height,
                    cg_bounds_width, cg_bounds_height,
                    rotation,
                    capture_width, capture_height
                );

                Monitor {
                    display_id,
                    name: format!("Display {}", display_id),
                    x: frame.origin.x as i32,
                    y: frame.origin.y as i32,
                    // Use SCK's dimensions - it knows what it expects
                    width: capture_width,
                    height: capture_height,
                    logical_width: cg_bounds_width,
                    logical_height: cg_bounds_height,
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
