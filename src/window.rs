//! Window capture using ScreenCaptureKit via cidre

use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionaryRef;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use core_graphics::window::{
    kCGNullWindowID, kCGWindowListOptionOnScreenOnly, kCGWindowListExcludeDesktopElements,
    CGWindowListCopyWindowInfo,
};
use image::RgbaImage;
use std::collections::HashMap;
use tracing::debug;

use crate::capture;
use crate::error::{XCapError, XCapResult};

/// Get window info from CGWindowList API as a fallback
/// Returns a HashMap of window_id -> (app_name, title, pid)
fn get_cgwindow_info() -> HashMap<u32, (String, String, i32)> {
    let mut info = HashMap::new();

    unsafe {
        let options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements;
        let window_list = CGWindowListCopyWindowInfo(options, kCGNullWindowID);

        if window_list.is_null() {
            return info;
        }

        let cf_array = core_foundation::array::CFArray::<CFType>::wrap_under_create_rule(
            window_list as core_foundation::array::CFArrayRef
        );
        let count = cf_array.len();

        for i in 0..count {
            if let Some(dict) = cf_array.get(i) {
                let dict_ref = dict.as_CFTypeRef() as CFDictionaryRef;

                // Get window ID
                let window_id = get_cf_number_value(dict_ref, "kCGWindowNumber").unwrap_or(-1);
                if window_id <= 0 {
                    continue;
                }

                // Get owner name (app name)
                let owner_name = get_cf_string_value(dict_ref, "kCGWindowOwnerName")
                    .unwrap_or_default();

                // Get window name (title)
                let window_name = get_cf_string_value(dict_ref, "kCGWindowName")
                    .unwrap_or_default();

                // Get owner PID
                let owner_pid = get_cf_number_value(dict_ref, "kCGWindowOwnerPID").unwrap_or(-1);

                info.insert(window_id as u32, (owner_name, window_name, owner_pid));
            }
        }
    }

    info
}

/// Get a string value from a CFDictionary
fn get_cf_string_value(dict: CFDictionaryRef, key: &str) -> Option<String> {
    unsafe {
        let cf_key = CFString::new(key);
        let mut value: *const std::ffi::c_void = std::ptr::null();

        let found = core_foundation::dictionary::CFDictionaryGetValueIfPresent(
            dict,
            cf_key.as_concrete_TypeRef() as *const _,
            &mut value,
        );
        if found != 0 && !value.is_null() {
            let cf_string = CFString::wrap_under_get_rule(value as core_foundation::string::CFStringRef);
            Some(cf_string.to_string())
        } else {
            None
        }
    }
}

/// Get a number value from a CFDictionary
fn get_cf_number_value(dict: CFDictionaryRef, key: &str) -> Option<i32> {
    unsafe {
        let cf_key = CFString::new(key);
        let mut value: *const std::ffi::c_void = std::ptr::null();

        let found = core_foundation::dictionary::CFDictionaryGetValueIfPresent(
            dict,
            cf_key.as_concrete_TypeRef() as *const _,
            &mut value,
        );
        if found != 0 && !value.is_null() {
            let cf_number = CFNumber::wrap_under_get_rule(value as core_foundation::number::CFNumberRef);
            cf_number.to_i32()
        } else {
            None
        }
    }
}

/// Represents a capturable window
///
/// This type provides an API compatible with xcap::Window
#[derive(Debug, Clone)]
pub struct Window {
    /// The window ID
    window_id: u32,
    /// The owning application name
    app_name: String,
    /// The window title
    title: String,
    /// Process ID of the owning application
    pid: i32,
    /// Window position X
    x: i32,
    /// Window position Y
    y: i32,
    /// Window width
    width: u32,
    /// Window height
    height: u32,
    /// Whether the window is on screen
    is_on_screen: bool,
}

impl Window {
    /// Get all available windows
    ///
    /// Returns a list of all windows that can be captured.
    /// Requires screen recording permission.
    pub fn all() -> XCapResult<Vec<Window>> {
        let content = capture::get_shareable_content()?;

        let sc_windows = content.windows();

        if sc_windows.is_empty() {
            return Err(XCapError::no_windows());
        }

        // Get CGWindow info as fallback for when SCK doesn't provide app metadata
        let cgwindow_info = get_cgwindow_info();

        let windows: Vec<Window> = sc_windows
            .iter()
            .filter_map(|w| {
                let window_id = w.id();

                // Get window properties from SCK first
                let title = w.title()
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                let (mut app_name, mut pid) = match w.owning_app() {
                    Some(app) => (app.app_name().to_string(), app.process_id()),
                    None => (String::new(), -1),
                };

                // Fallback to CGWindow API if SCK didn't provide app info
                if app_name.is_empty() || pid < 0 {
                    if let Some((cg_app_name, _cg_title, cg_pid)) = cgwindow_info.get(&window_id) {
                        if app_name.is_empty() && !cg_app_name.is_empty() {
                            debug!("Using CGWindow fallback for app_name: {} -> {}", window_id, cg_app_name);
                            app_name = cg_app_name.clone();
                        }
                        if pid < 0 && *cg_pid >= 0 {
                            pid = *cg_pid;
                        }
                    }
                }

                // Get window frame
                let frame = w.frame();
                let width = frame.size.width as u32;
                let height = frame.size.height as u32;

                // Skip windows that are too small (likely invisible)
                if width < 10 || height < 10 {
                    debug!("Skipping small window: {} ({}x{})", title, width, height);
                    return None;
                }

                debug!(
                    "Found window: id={}, app={}, title={}, {}x{} at ({}, {})",
                    window_id, app_name, title, width, height, frame.origin.x, frame.origin.y
                );

                Some(Window {
                    window_id,
                    app_name,
                    title,
                    pid,
                    x: frame.origin.x as i32,
                    y: frame.origin.y as i32,
                    width,
                    height,
                    is_on_screen: w.is_on_screen(),
                })
            })
            .collect();

        if windows.is_empty() {
            return Err(XCapError::no_windows());
        }

        Ok(windows)
    }

    /// Get the window ID
    pub fn id(&self) -> XCapResult<u32> {
        Ok(self.window_id)
    }

    /// Get the window's raw ID (non-Result version for convenience)
    pub fn raw_id(&self) -> u32 {
        self.window_id
    }

    /// Get the window's process ID
    pub fn pid(&self) -> XCapResult<u32> {
        if self.pid < 0 {
            return Err(XCapError::new("Process ID not available"));
        }
        Ok(self.pid as u32)
    }

    /// Get the application name
    pub fn app_name(&self) -> XCapResult<String> {
        Ok(self.app_name.clone())
    }

    /// Get the window title
    pub fn title(&self) -> XCapResult<String> {
        Ok(self.title.clone())
    }

    /// Get the window X position
    pub fn x(&self) -> XCapResult<i32> {
        Ok(self.x)
    }

    /// Get the window Y position
    pub fn y(&self) -> XCapResult<i32> {
        Ok(self.y)
    }

    /// Get the window width
    pub fn width(&self) -> XCapResult<u32> {
        Ok(self.width)
    }

    /// Get the window height
    pub fn height(&self) -> XCapResult<u32> {
        Ok(self.height)
    }

    /// Check if the window is minimized
    pub fn is_minimized(&self) -> XCapResult<bool> {
        // SCK provides is_on_screen which is the inverse
        Ok(!self.is_on_screen)
    }

    /// Check if the window is maximized
    pub fn is_maximized(&self) -> XCapResult<bool> {
        // TODO: Compare with monitor size
        Ok(false)
    }

    /// Check if the window is focused
    ///
    /// Note: ScreenCaptureKit doesn't directly provide focus state.
    /// Returns true if the window is on screen as a reasonable proxy.
    pub fn is_focused(&self) -> XCapResult<bool> {
        // Use is_on_screen as proxy for focus since SCK doesn't expose focus state
        // This allows screenpipe to capture on-screen windows
        Ok(self.is_on_screen)
    }

    /// Check if the window is on screen
    pub fn is_on_screen(&self) -> bool {
        self.is_on_screen
    }

    /// Capture an image of the window
    ///
    /// Returns an RGBA image of the window contents.
    pub fn capture_image(&self) -> XCapResult<RgbaImage> {
        capture::capture_window_sync(self.window_id, self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_getters() {
        let window = Window {
            window_id: 123,
            app_name: "TestApp".to_string(),
            title: "Test Window".to_string(),
            pid: 456,
            x: 100,
            y: 200,
            width: 800,
            height: 600,
            is_on_screen: true,
        };

        assert_eq!(window.id().unwrap(), 123);
        assert_eq!(window.raw_id(), 123);
        assert_eq!(window.app_name().unwrap(), "TestApp");
        assert_eq!(window.title().unwrap(), "Test Window");
        assert_eq!(window.pid().unwrap(), 456);
        assert_eq!(window.x().unwrap(), 100);
        assert_eq!(window.y().unwrap(), 200);
        assert_eq!(window.width().unwrap(), 800);
        assert_eq!(window.height().unwrap(), 600);
        assert!(!window.is_minimized().unwrap());
        assert!(window.is_on_screen());
    }

    #[test]
    fn test_window_minimized() {
        let window = Window {
            window_id: 1,
            app_name: "App".to_string(),
            title: "Title".to_string(),
            pid: 1,
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            is_on_screen: false, // Not on screen = minimized
        };

        assert!(window.is_minimized().unwrap());
        assert!(!window.is_on_screen());
    }

    #[test]
    fn test_window_all() {
        // This test verifies the API works
        // It will fail or succeed based on permission state
        let result = Window::all();
        // We just check it returns a result, not panics
        let _ = result;
    }
}
