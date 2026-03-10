#!/usr/bin/env python3
"""
Barcode / QR Code Scanner
==========================
Opens the camera at high resolution and continuously scans for
barcodes and QR codes in real time — no button press needed.

Scanning runs in a background thread so the display stays fluid.
The camera captures at 1920×1080 (MJPG) and the window is scaled
to fit HiDPI screens.

Usage:
    python read_barcode.py                # live camera mode
    python read_barcode.py <image_path>   # process a single image
    python read_barcode.py --copy         # auto-copy last scan to clipboard

Keys (camera mode):
    S        -> save current frame as image
    C        -> copy last scanned value to clipboard
    R        -> reset scan history
    ESC / Q  -> quit
"""

import sys
import time
import threading
import subprocess
import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as decode_barcodes, ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("ERROR: pyzbar is required. Install it:")
    print("  pip install pyzbar")
    print("  sudo dnf install -y zbar  # (or apt install libzbar0)")
    sys.exit(1)

AUTO_COPY = "--copy" in sys.argv

# Barcode symbol types to scan (excluding PDF417 — very slow & noisy)
SCAN_SYMBOLS = [
    ZBarSymbol.QRCODE,
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
    ZBarSymbol.CODE128,
    ZBarSymbol.CODE39,
    ZBarSymbol.CODE93,
    ZBarSymbol.I25,       # Interleaved 2 of 5
    ZBarSymbol.DATABAR,
    ZBarSymbol.CODABAR,
    ZBarSymbol.ISBN13,
    ZBarSymbol.ISBN10,
]

# Width to downscale frames to before running pyzbar (for speed)
SCAN_WIDTH = 640

# Display window target width (fits well on HiDPI ~2880px wide screens)
DISPLAY_WIDTH = 1200

# ── Colours ───────────────────────────────────────────────────────────

GREEN   = (0, 255, 0)
YELLOW  = (0, 255, 255)
RED     = (0, 0, 255)
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
DARK_BG = (40, 40, 40)

# ── Clipboard helper ─────────────────────────────────────────────────

def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using wl-copy (Wayland) or xclip fallback."""
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False

# ── Barcode scanning ─────────────────────────────────────────────────

def scan_barcodes(frame: np.ndarray, scale_coords: float = 1.0) -> list[dict]:
    """Detect barcodes/QR codes in a frame.

    Runs on a downscaled greyscale image for speed.  Polygon and rect
    coordinates are scaled back to the original frame size using
    *scale_coords*.  Early-exits once the first variant finds results.
    """
    results: list[dict] = []
    seen_data: set[str] = set()

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Lazy variant generators — only computed if previous ones found nothing
    def _variants():
        yield gray                                                        # greyscale
        yield cv2.equalizeHist(gray)                                      # histogram eq
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yield clahe.apply(gray)                                           # CLAHE
        yield cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]       # Otsu

    for variant in _variants():
        try:
            decoded = decode_barcodes(variant, symbols=SCAN_SYMBOLS)
        except Exception:
            continue

        for bc in decoded:
            data = bc.data.decode("utf-8", errors="replace")
            key = f"{bc.type}:{data}"
            if key not in seen_data:
                seen_data.add(key)
                s = scale_coords
                pts = [(int(p.x * s), int(p.y * s))
                       for p in bc.polygon] if bc.polygon else []
                r = bc.rect
                results.append({
                    "type": bc.type,
                    "data": data,
                    "polygon": pts,
                    "rect": (int(r.left * s), int(r.top * s),
                             int(r.width * s), int(r.height * s)),
                })

        if results:          # early exit — no need to try more variants
            break

    return results


# ── Drawing helpers ───────────────────────────────────────────────────

def draw_barcode_overlay(frame: np.ndarray, barcodes: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels for each detected barcode."""
    overlay = frame.copy()

    for bc in barcodes:
        pts = bc["polygon"]
        rx, ry, rw, rh = bc["rect"]
        label = f'{bc["type"]}: {bc["data"]}'

        # Draw polygon outline (green)
        if len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [poly], isClosed=True, color=GREEN, thickness=3)
        else:
            # Fallback to bounding rectangle
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), GREEN, 3)

        # Semi-transparent label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(ry - 10, th + 6)
        cv2.rectangle(overlay, (rx, label_y - th - 6), (rx + tw + 10, label_y + 4), BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
        # Re-draw the label on top (after blending)
        cv2.putText(overlay, label, (rx + 5, label_y - 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        # Re-draw polygon so it stays sharp after blending
        if len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [poly], isClosed=True, color=GREEN, thickness=3)
        else:
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), GREEN, 3)

    return overlay


def draw_hud(frame: np.ndarray, history: list[dict], status: str, fps: float) -> np.ndarray:
    """Draw heads-up display: status bar, FPS, and scan history."""
    h, w = frame.shape[:2]
    display = frame.copy()

    # ── Top bar (FPS) ──
    cv2.rectangle(display, (0, 0), (w, 32), DARK_BG, -1)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    cv2.putText(display, f"Scanned: {len(history)}", (w - 180, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)

    # ── Bottom bar (status) ──
    cv2.rectangle(display, (0, h - 36), (w, h), DARK_BG, -1)
    cv2.putText(display, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # ── Scan history panel (last 6, right side) ──
    if history:
        panel_lines = history[-6:]
        panel_h = len(panel_lines) * 28 + 16
        panel_w = min(w // 2, 500)
        y0 = 40
        # Semi-transparent background
        sub = display[y0:y0 + panel_h, w - panel_w - 10:w - 5]
        if sub.size > 0:
            dark = np.zeros_like(sub)
            cv2.addWeighted(sub, 0.4, dark, 0.6, 0, sub)
            display[y0:y0 + panel_h, w - panel_w - 10:w - 5] = sub

        for i, entry in enumerate(panel_lines):
            txt = f'{entry["type"]}: {entry["data"]}'
            if len(txt) > 55:
                txt = txt[:52] + "..."
            ty = y0 + 22 + i * 28
            cv2.putText(display, txt, (w - panel_w, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    return display


# ── Camera resolution setup ───────────────────────────────────────────

def configure_camera(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Try to set the highest practical resolution for barcode scanning.
    
    1920×1080 is the sweet spot: high enough detail for small barcodes,
    fast enough for real-time processing.  Falls back to 1280×720.
    """
    # Preferred resolutions in descending order
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (960, 540),
    ]

    # Prefer MJPEG backend for higher resolution support
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w >= w and actual_h >= h:
            break

    # Autofocus on (helps barcode clarity)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    final_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return final_w, final_h


# ── Process a single image ────────────────────────────────────────────

def process_image(path: str) -> None:
    """Load an image, scan for barcodes, and print results."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not read '{path}'")
        sys.exit(1)

    print(f"Processing: {path}")
    barcodes = scan_barcodes(img)

    if not barcodes:
        print("  No barcodes found.")
        return

    print(f"  Found {len(barcodes)} barcode(s):\n")
    for i, bc in enumerate(barcodes, 1):
        print(f"  [{i}]  Type: {bc['type']}")
        print(f"       Data: {bc['data']}")
        print()

    # Show the image with overlays
    annotated = draw_barcode_overlay(img, barcodes)
    cv2.imshow("Barcode Scanner — Result", annotated)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Background scanner thread ─────────────────────────────────────────

class BarcodeScanner:
    """Runs barcode scanning in a background thread so the UI stays fluid."""

    def __init__(self):
        self.lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._results: list[dict] = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Hand a new frame to the scanner (non-blocking, drops old ones)."""
        with self.lock:
            self._latest_frame = frame

    def get_results(self) -> list[dict]:
        """Return the most recent scan results (thread-safe)."""
        with self.lock:
            return list(self._results)

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            # Grab the latest frame
            with self.lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.005)      # idle — avoid busy-waiting
                continue

            # Downscale for speed
            h, w = frame.shape[:2]
            if w > SCAN_WIDTH:
                scale = SCAN_WIDTH / w
                small = cv2.resize(frame, (SCAN_WIDTH, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
                coord_scale = w / SCAN_WIDTH    # scale coords back up
            else:
                small = frame
                coord_scale = 1.0

            results = scan_barcodes(small, scale_coords=coord_scale)

            with self.lock:
                self._results = results


# ── Live camera loop ──────────────────────────────────────────────────

def camera_loop() -> None:
    """Open the camera and continuously scan for barcodes in real time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        sys.exit(1)

    cam_w, cam_h = configure_camera(cap)

    # Compute display size that fits nicely on HiDPI screens
    display_scale = DISPLAY_WIDTH / cam_w
    disp_w = DISPLAY_WIDTH
    disp_h = int(cam_h * display_scale)

    print("+" + "=" * 54 + "+")
    print("|  BARCODE / QR SCANNER — REAL-TIME                    |")
    print("|                                                      |")
    print(f"|  Camera : {cam_w}x{cam_h}".ljust(55) + "|")
    print(f"|  Window : {disp_w}x{disp_h} (scaled for HiDPI)".ljust(55) + "|")
    print("|  Barcodes are detected automatically.                |")
    print("|                                                      |")
    print("|  S        -> save frame as image                     |")
    print("|  C        -> copy last scan to clipboard             |")
    print("|  R        -> reset history                           |")
    print("|  ESC / Q  -> quit                                    |")
    print("+" + "=" * 54 + "+\n")

    # Window setup — WINDOW_NORMAL lets us control the size on HiDPI
    win_name = "Barcode Scanner"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)

    scanner = BarcodeScanner()

    history: list[dict] = []          # all unique scans this session
    history_keys: set[str] = set()    # for dedup
    last_scan_value: str = ""
    status = "Scanning... point camera at a barcode"
    cooldown_until: float = 0.0       # avoid spamming same code

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            break

        # Hand the frame to the background scanner
        scanner.submit(frame)

        # Grab latest results (non-blocking)
        barcodes = scanner.get_results()
        display = frame

        if barcodes:
            display = draw_barcode_overlay(frame, barcodes)

            now = time.time()
            for bc in barcodes:
                key = f'{bc["type"]}:{bc["data"]}'

                # New unique barcode or cooldown expired for repeated scan
                if key not in history_keys or now >= cooldown_until:
                    if key not in history_keys:
                        history.append({"type": bc["type"], "data": bc["data"]})
                        history_keys.add(key)

                    last_scan_value = bc["data"]
                    cooldown_until = now + 1.5  # 1.5 s cooldown per code

                    # Print to terminal
                    print(f"  [{bc['type']}]  {bc['data']}")

                    # Auto-copy if requested
                    if AUTO_COPY:
                        if copy_to_clipboard(bc["data"]):
                            status = f"Copied: {bc['data'][:40]}"
                        else:
                            status = f"Scanned: {bc['data'][:40]} (clipboard unavailable)"
                    else:
                        status = f"Scanned: {bc['data'][:50]}"

        # ── FPS calculation ──
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # ── HUD ──
        display = draw_hud(display, history, status, fps)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            fname = f"barcode_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  Frame saved: {fname}")
            status = f"Saved: {fname}"
        elif key in (ord("c"), ord("C")):
            if last_scan_value:
                if copy_to_clipboard(last_scan_value):
                    print(f"  Copied to clipboard: {last_scan_value}")
                    status = f"Copied: {last_scan_value[:40]}"
                else:
                    print("  Clipboard copy failed (install wl-copy or xclip)")
                    status = "Clipboard copy failed"
            else:
                status = "Nothing to copy yet"
        elif key in (ord("r"), ord("R")):
            history.clear()
            history_keys.clear()
            last_scan_value = ""
            status = "History cleared — scanning..."
            print("  History reset.")

    # ── Cleanup ──
    scanner.stop()
    cap.release()
    cv2.destroyAllWindows()

    # Print session summary
    if history:
        print("\n" + "=" * 50)
        print("  SESSION SUMMARY")
        print("=" * 50)
        for i, entry in enumerate(history, 1):
            print(f"  [{i}]  {entry['type']}: {entry['data']}")
        print("=" * 50 + "\n")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        process_image(args[0])
    else:
        camera_loop()


if __name__ == "__main__":
    main()
