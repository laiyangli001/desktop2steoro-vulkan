from __future__ import annotations

import os
import subprocess
import threading
import time

import glfw

from utils import (
    AUDIO_DELAY,
    CRF,
    FPS,
    LOSSLESS_SCALING_SUPPORT,
    OS_NAME,
    STEREOMIX_DEVICE,
    STREAM_KEY,
    shutdown_event,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTMP_DIR = os.path.join(BASE_DIR, "streaming", "rtmp")

global_processes = {
    "ffmpeg": None,
    "rtmp_server": None,
}
current_stream_size = None
ffmpeg_restart_lock = threading.Lock()

def get_rtmp_cmd(os_name=OS_NAME, window=None):
    if not window:
        raise ValueError("GLFW window required for size-aware streaming")
    
    time.sleep(0.5) # wait 0.5 seconds to get correct window position
    width, height = glfw.get_framebuffer_size(window)
    if width > 0 and height > 0:
        width = (width // 2) * 2  # make even
        height = (height // 2) * 2  # make even
    else:
        time.sleep(3) # retry after 3 seconds
        get_rtmp_cmd()
        
    if os_name == "Windows":
        server_cmd = [os.path.join(RTMP_DIR, "mediamtx", "mediamtx.exe"), os.path.join(RTMP_DIR, "mediamtx", "mediamtx.yml")]
        if not LOSSLESS_SCALING_SUPPORT: 
            ffmpeg_cmd = [
                os.path.join(RTMP_DIR, "ffmpeg", "bin", "ffmpeg.exe"),
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '64',
                '-analyzeduration', '0',
                '-filter_complex', f"gfxcapture=window_title='(?i)Stereo Viewer':capture_cursor=0:max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",  # Label video output [v], fix odd height
                '-itsoffset', f'{AUDIO_DELAY}',  # Audio delay (applies to next input)
                '-f', 'dshow',
                '-rtbufsize', '256M',
                '-i', f'audio={STEREOMIX_DEVICE}',
                '-map', '[v]',
                '-map', '0:a',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-bf', '0',
                '-g', f'{FPS}',
                '-force_key_frames', f'expr:gte(t,n_forced*1)',  # Force keyframes every second
                '-r', f'{FPS}',  # Force constant output framerate
                '-crf', f'{CRF}', # 18-24 smaller better quality
                '-c:a', 'libopus',
                # '-ar', '44100',
                '-b:a', '96k',
                '-muxdelay', '0',
                '-muxpreload', '0',
                '-flush_packets', '1',
                '-f', 'mpegts',
                f'srt://localhost:8890?streamid=publish:{STREAM_KEY}&pkt_size=1316'
            ]
        else:
            import win32gui
            import win32api

            def find_window_by_prefix(prefix):
                target_hwnd = None
                def enum_handler(hwnd, _):
                    nonlocal target_hwnd
                    title = win32gui.GetWindowText(hwnd)
                    if title.lower().startswith(prefix.lower()) and win32gui.IsWindowVisible(hwnd):
                        target_hwnd = hwnd
                win32gui.EnumWindows(enum_handler, None)
                return target_hwnd

            def get_monitor_index_for_window(hwnd):
                """Returns the 0-based monitor index (matching gfxcapture order) that contains the center of the window"""
                if not hwnd:
                    raise ValueError("Invalid window handle")
                
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                monitors = []
                for hMonitor, hdcMonitor, (left, top, right, bottom) in win32api.EnumDisplayMonitors():
                    info = win32api.GetMonitorInfo(hMonitor)
                    monitors.append({
                        'index': len(monitors),  # This order matches gfxcapture monitor_idx
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'width': right - left,
                        'height': bottom - top,
                    })
                
                for mon in monitors:
                    if mon['left'] <= center_x < mon['right'] and mon['top'] <= center_y < mon['bottom']:
                        return mon['index'], mon
                
                raise RuntimeError("Could not determine monitor for window")

            # Main logic
            hwnd = find_window_by_prefix("Stereo Viewer")
            if not hwnd:
                raise RuntimeError("Window starting with 'Stereo Viewer' not found")

            # Get monitor index and full monitor rect
            monitor_idx, monitor = get_monitor_index_for_window(hwnd)

            # Get window client area (recommended: excludes title bar and borders)
            window_left, window_top = win32gui.ClientToScreen(hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(hwnd)
            window_right = window_left + client_rect[2]
            window_bottom = window_top + client_rect[3]

            # Alternatively: full window including borders/title bar
            # window_left, window_top, window_right, window_bottom = win32gui.GetWindowRect(hwnd)

            print(f"[OK]Stereo Viewer window found on monitor {monitor_idx}")
            print(f"Monitor bounds: X={monitor['left']} Y={monitor['top']} W={monitor['width']} H={monitor['height']}")
            print(f"Window client area: X={window_left} Y={window_top} -> {window_right}x{window_bottom}")

            # Calculate crop values so that the captured monitor region is cropped exactly to the window
            crop_left   = window_left - monitor['left']
            crop_top    = window_top - monitor['top']
            crop_right  = monitor['right'] - window_right
            crop_bottom = monitor['bottom'] - window_bottom

            # Ensure non-negative crops (in case window is partially off-screen or miscalculated)
            crop_left   = max(0, crop_left)
            crop_top    = max(0, crop_top)
            crop_right  = max(0, crop_right)
            crop_bottom = max(0, crop_bottom)

            print("Calculated crop values (to capture only the window area):")
            print(f"crop_left   = {crop_left}")
            print(f"crop_top    = {crop_top}")
            print(f"crop_right  = {crop_right}")
            print(f"crop_bottom = {crop_bottom}")

            # Build gfxcapture options
            gfxcapture_options = f"monitor_idx={monitor_idx}:crop_left={crop_left}:crop_top={crop_top}:crop_right={crop_right}:crop_bottom={crop_bottom}"

            ffmpeg_cmd = [
                os.path.join(RTMP_DIR, "ffmpeg", "bin", "ffmpeg.exe"),
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '64',
                '-analyzeduration', '0',
                '-filter_complex', f"gfxcapture={gfxcapture_options}:capture_cursor=0:max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",
                '-itsoffset', f'{AUDIO_DELAY}',
                '-f', 'dshow',
                '-rtbufsize', '256M',
                '-i', f'audio={STEREOMIX_DEVICE}',
                '-map', '[v]',
                '-map', '0:a',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-bf', '0',
                '-g', f'{FPS}',
                '-force_key_frames', f'expr:gte(t,n_forced*1)',
                '-r', f'{FPS}',
                '-crf', f'{CRF}',
                '-c:a', 'libopus',
                '-b:a', '96k',
                '-muxdelay', '0',
                '-muxpreload', '0',
                '-flush_packets', '1',
                '-f', 'mpegts',
                f'srt://localhost:8890?streamid=publish:{STREAM_KEY}&pkt_size=1316'
            ]
            
    elif os_name == "Darwin":
        
        from AppKit import NSScreen

        def get_scale(monitor_index):
            """Get the Retina scale factor for a specific monitor"""
            screens = NSScreen.screens()
            if monitor_index < len(screens):
                return screens[monitor_index].backingScaleFactor()
            return 2.0  # Default to 2x for Retina displays
        
        def get_monitor_index_for_glfw(window):
            window_x, window_y = glfw.get_window_pos(window)
            monitors = glfw.get_monitors()

            for i, monitor in enumerate(monitors):
                mx, my = glfw.get_monitor_pos(monitor)
                mode = glfw.get_video_mode(monitor)
                mw, mh = mode.size.width, mode.size.height

                if mx <= window_x < mx + mw and my <= window_y < my + mh:
                    return i
            return 0
        import re
        def get_device_index(target_name, device_type="video"):
            cmd = [os.path.join(RTMP_DIR, "mac", "ffmpeg"), "-f", "avfoundation", "-list_devices", "true", "-i", ""]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            output = result.stderr

            found_audio = False
            for line in output.splitlines():
                if "AVFoundation audio devices:" in line:
                    found_audio = True
                if "AVFoundation video devices:" in line:
                    found_audio = False

                match = re.search(r'\[(\d+)\](.+)', line)
                if match:
                    index = int(match.group(1))
                    name = match.group(2).strip()
                    if name == target_name and ((device_type == "audio" and found_audio) or (device_type == "video" and not found_audio)):
                        return index
            return None
        monitor_index = get_monitor_index_for_glfw(window)
        monitors = glfw.get_monitors()
        monitor = monitors[monitor_index]
        scale_factor = get_scale(monitor_index)
        screen_name = f"Capture screen {monitor_index}"
        screen_index = get_device_index(screen_name, "video")
        audio_index = get_device_index(STEREOMIX_DEVICE, "audio")
        win_x, win_y = glfw.get_window_pos(window)
        mon_x, mon_y = glfw.get_monitor_pos(monitor)
        x = win_x - mon_x
        y = win_y - mon_y
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        
        server_cmd = [os.path.join(RTMP_DIR, "mac", "mediamtx"), os.path.join(RTMP_DIR, "mac", "mediamtx.yml")]
        ffmpeg_cmd = [
            os.path.join(RTMP_DIR, "mac", "ffmpeg"),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "1024",
            "-analyzeduration", "0",
            "-itsoffset", str(AUDIO_DELAY),
            "-pixel_format", "uyvy422",
            "-f", "avfoundation",
            "-rtbufsize", "256M",
            "-framerate", "60",
            "-i", f"{screen_index}:{audio_index}",
            "-filter_complex",
            f"[0:v]fps={FPS},crop={width}:{height}:{x}:{y},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=uyvy422[v];[0:a]aresample=async=1[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "h264_videotoolbox",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-b:v", "10M",
            "-maxrate", "12M",
            "-bufsize", "24M",
            "-g", str(FPS),
            "-r", str(FPS),
            "-realtime", "true",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-c:a", "libopus",
            "-b:a", "96k",
            "-ar", "48000",
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}",
        ]


    elif os_name == "Linux":
        import re

        def run(cmd):
            """Run a shell command and return output as string."""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip()

        def get_window_geometry(window_id, entire=True):
            """
            Parse xwininfo to get full window geometry and decorations (border + title bar).
            Returns: (x, y, width, height, border, titlebar)
            """
            output = run(f"xwininfo -id {window_id}")
            x = y = w = h = b = t = 0

            # Parse lines similarly to the sed version
            for line in output.splitlines():
                line = line.strip()
                if m := re.match(r"Absolute upper-left X:\s+(\d+)", line):
                    x = int(m.group(1))
                elif m := re.match(r"Absolute upper-left Y:\s+(\d+)", line):
                    y = int(m.group(1))
                elif m := re.match(r"Width:\s+(\d+)", line):
                    w = int(m.group(1))
                elif m := re.match(r"Height:\s+(\d+)", line):
                    h = int(m.group(1))
                elif m := re.match(r"Relative upper-left X:\s+(\d+)", line):
                    b = int(m.group(1))
                elif m := re.match(r"Relative upper-left Y:\s+(\d+)", line):
                    t = int(m.group(1))

            # Adjust if user wanted entire window including borders/titlebar
            if entire:
                x -= b
                y -= t
                w += 2 * b
                h += t + b

            return x, y, w, h, b, t

        def drag_window_offscreen(window_id, dx=10000, dy=10000, steps=1, delay=0.01):
            """
            Simulate a mouse drag on a window by ID using xdotool.
            Uses xwininfo to compute exact title bar position (no scale factor needed).
            """
            # Get window geometry + decoration info
            x, y, w, h, b, t = get_window_geometry(window_id)
            print(f"Window pos=({x},{y}), size={w}x{h}, border={b}, title={t}")

            # Activate the window
            run(f"xdotool windowactivate {window_id}")

            # Compute title bar click position
            # title_x = x + w // 2
            title_x = x + 20
            title_y = y + t // 2  # halfway down the title bar for reliable click

            # Move mouse to title bar and start drag
            run(f"xdotool mousemove {title_x} {title_y}")
            run("xdotool mousedown 1")

            # Smooth drag motion
            step_x = dx / steps
            step_y = dy / steps
            for _ in range(steps):
                run(f"xdotool mousemove_relative -- {step_x:.2f} {step_y:.2f}")
                time.sleep(delay)

            # Release mouse button
            run("xdotool mouseup 1")



        def get_display_env(window_id: str) -> str:
            """
            Get the DISPLAY environment variable from the process that owns the window.
            Falls back to ':0.0' if not found or on error.
            """
            try:
                # First, get the PID of the window
                pid_result = subprocess.run(
                    ["xprop", "-id", window_id, "_NET_WM_PID"],
                    capture_output=True, text=True, check=True
                )
                
                # Parse PID from output (e.g., '_NET_WM_PID(CARDINAL) = 1234')
                for line in pid_result.stdout.splitlines():
                    if "=" in line:
                        pid = line.split("=")[-1].strip()
                        if pid.isdigit():
                            # Get environment variables from the process
                            env_result = subprocess.run(
                                ["cat", f"/proc/{pid}/environ"],
                                capture_output=True, text=True, check=True
                            )
                            
                            # Parse DISPLAY from environment variables
                            for env_var in env_result.stdout.split('\x00'):
                                if env_var.startswith("DISPLAY="):
                                    return env_var.split("=", 1)[1]
                                    
            except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
                pass
            
            return ":0.0"

        def get_device_index(target_name, device_type="video"):
            """Return a device identifier suitable for ffmpeg on Linux.
            For video: returns a display/input string (we'll use it later as the ffmpeg input).
            For audio: tries to locate a PulseAudio/ALSA name that matches target_name; returns 'default' if not found.
            """
            # AUDIO: try PulseAudio device listing via ffmpeg
            if device_type == "audio":
                # Use the bundled or system ffmpeg binary; fallback to system
                cmd = ["ffmpeg", "-f", "pulse", "-list_devices", "true", "-i", ""]  # PulseAudio list (printed to stderr)
                try:
                    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    output = result.stderr + result.stdout
                    # ffmpeg prints devices with lines like: "    name 'alsa_input.pci-0000_00_1b.0.analog-stereo'": device names can appear
                    # Simpler: find lines containing the target_name or the friendly name
                    for line in output.splitlines():
                        if target_name in line:
                            # extract quoted name if present
                            m = re.search(r"'([^']+)'", line)
                            if m:
                                return m.group(1)
                            # otherwise attempt to extract a token
                            tokens = line.strip().split()
                            if tokens:
                                return tokens[-1].strip()
                except Exception:
                    pass

                # Try listing via pactl as fallback
                try:
                    pactl_out = subprocess.check_output(["pactl", "list", "sources"], text=True)
                    # parse "Name: <name>" and "Description: <desc>"
                    name = None
                    for line in pactl_out.splitlines():
                        line = line.strip()
                        if line.startswith("Name:"):
                            name = line.split(":", 1)[1].strip()
                        if line.startswith("Description:") and target_name in line:
                            return name or "default"
                except Exception:
                    pass

                # fallback
                return "default"
            return target_name or ":0.0"
        
        def find_window_id_by_title(title_pattern: str) -> str | None:
            """
            Returns the hex window id (e.g. '0x4a0000b') of the first window whose
            title contains *title_pattern* (case-insensitive).  Uses `xwininfo -tree -root`.
            """
            try:
                out = subprocess.check_output(
                    ["xwininfo", "-root", "-tree"], text=True, stderr=subprocess.DEVNULL
                )
                # Example line:
                #     0x4a0000b "Stereo Viewer - Left Eye": ("Stereo Viewer - Left Eye" ...
                pat = re.compile(
                    rf'^\s+(0x[0-9a-fA-F]+)\s.*?"[^"]*{re.escape(title_pattern)}[^"]*"',
                    re.IGNORECASE,
                )
                for line in out.splitlines():
                    m = pat.match(line)
                    if m:
                        return m.group(1)
            except Exception as e:
                print(f"[window_id] search failed: {e}")
            return None

        # Make sure the GLFW window has a recognizable title:
        glfw_title = glfw.get_window_title(window) or ""
        search_title = glfw_title if "Stereo Viewer" in glfw_title else "Stereo Viewer"

        window_id = find_window_id_by_title(search_title)
        display_env = get_display_env(window_id)+".0"
        
        if not window_id:
            raise RuntimeError(
                f"Could not locate a window with title containing '{search_title}'. "
                "Check glfw.set_window_title() or run `xwininfo -tree -root | grep -i stereo`."
            )
        print(f"[info] Capturing window id {window_id}")
        drag_window_offscreen(window_id)

        # audio_index = get_device_index(STEREOMIX_DEVICE, "audio")
        server_cmd = [os.path.join(RTMP_DIR, "linux", "mediamtx"), os.path.join(RTMP_DIR, "linux", "mediamtx.yml")]

        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "+genpts+nobuffer+flush_packets",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-probesize", "1024",
            "-analyzeduration", "0",
            "-draw_mouse", "0",
            "-itsoffset", str(AUDIO_DELAY),
            "-f", "x11grab",
            "-framerate", "60", 
            "-vsync", "1",
            "-window_id", window_id,
            "-use_wallclock_as_timestamps", "1",
            "-thread_queue_size", "2048",
            "-i", display_env,
            "-f", "pulse",
            "-thread_queue_size", "512",
            "-i", STEREOMIX_DEVICE,
            "-ac", "2",
            "-filter_complex", f"[0:v]fps={FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,setpts=PTS-STARTPTS[v];[1:a]aresample=async=1:first_pts=0,apad[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-x264-params", f"keyint={FPS}:min-keyint={FPS}:scenecut=0:rc-lookahead=0",
            "-g", str(FPS),
            "-r", str(FPS),
            "-crf", str(CRF),
            "-c:a", "aac", 
            "-ar", "44100", 
            "-b:a", "96k", 
            "-threads", "2",
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}",
        ]

    return server_cmd, ffmpeg_cmd

# ffmpeg based rtmp streamer
def rtmp_stream(window, is_window_visible_on_screen):
    global current_stream_size

    # Wait for GLFW window to be fully initialized
    print("[RTMP] Waiting for window to be ready...")
    max_attempts = 100  # 5 seconds with 0.1s intervals
    for _ in range(max_attempts):
        if shutdown_event.is_set():
            return
        
        # Check if window is valid and has a size
        try:
            width, height = glfw.get_framebuffer_size(window)
            if width > 0 and height > 0 and is_window_visible_on_screen("Stereo Viewer"):
                print(f"[RTMP] Window ready: {width}x{height}")
                time.sleep(2)
                break
        except Exception as e:
            print(f"[RTMP] Window check error: {e}")
        
        time.sleep(0.5)
    
    if shutdown_event.is_set():
        return

    while not shutdown_event.is_set():
        try:
            width, height = glfw.get_framebuffer_size(window)
            new_size = (width, height)

            # Debounce: ignore tiny changes
            if current_stream_size and abs(current_stream_size[0] - new_size[0]) < 8 and abs(current_stream_size[1] - new_size[1]) < 8:
                time.sleep(0.1)
                continue

            with ffmpeg_restart_lock:
                if current_stream_size == new_size:
                    time.sleep(0.1)
                    continue
                current_stream_size = new_size

            # Terminate old processes
            for name in ['ffmpeg', 'rtmp_server']:
                proc = global_processes.get(name)
                if proc and proc.poll() is None:
                    print(f"[RTMP] Stopping {name} for resize...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()

            # Start new
            server_cmd, ffmpeg_cmd = get_rtmp_cmd(OS_NAME, window=window)
            print(f"[RTMP] Restarting stream at {width}x{height}")

            rtmp_server = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

            global_processes['rtmp_server'] = rtmp_server
            global_processes['ffmpeg'] = ffmpeg

            print(f"[RTMP] Stream active: {width}x{height}")

        except Exception as e:
            print(f"[RTMP] Error: {e}")
            time.sleep(1)

        time.sleep(0.2)

    print("[RTMP] Stream thread exited.")

