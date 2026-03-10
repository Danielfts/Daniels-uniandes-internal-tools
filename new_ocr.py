#!/usr/bin/env python3
"""
Uniandes Student Code — Real-Time OCR Reader
==============================================
Opens the camera at the highest available resolution and shows a
guide box for the user to centre the ID card.  OCR runs only on the
cropped guide region in a background thread, looking for the 9-digit
university student code (starts with 202…).

Usage:
    python new_ocr.py                  # live camera mode
    python new_ocr.py <image_path>     # process a single image
    python new_ocr.py --debug          # show all OCR word boxes

Keys (camera mode):
    SPACE    -> force an immediate OCR scan
    C        -> copy student code to clipboard
    S        -> save current frame
    D        -> toggle debug word boxes
    ESC / Q  -> quit
"""

import sys
import re
import time
import threading
import subprocess
import cv2
import numpy as np
import pytesseract

DEBUG = "--debug" in sys.argv

# ── Display ───────────────────────────────────────────────────────────

DISPLAY_WIDTH = 1200          # window width for HiDPI (2880×1800)

# Guide box: portrait orientation for holding the ID card upright
# Standard ID card is ~85.6 × 53.98 mm → portrait w/h ≈ 0.63
GUIDE_H_FRAC = 0.80           # 80 % of the frame height
GUIDE_ASPECT = 53.98 / 85.6   # w/h ratio in portrait (≈ 0.63)

# ── Colours (BGR) ─────────────────────────────────────────────────────

GREEN    = (0, 220, 0)
CYAN     = (220, 220, 0)
WHITE    = (255, 255, 255)
BLACK    = (0, 0, 0)
DARK_BG  = (35, 35, 35)
YELLOW   = (0, 255, 255)
GREY     = (180, 180, 180)
DIM_GREY = (120, 120, 120)

# ── Clipboard ─────────────────────────────────────────────────────────

def copy_to_clipboard(text: str) -> bool:
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False

# ── Camera setup ──────────────────────────────────────────────────────

def configure_camera(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Set the highest resolution the camera supports."""
    resolutions = [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1280,  720),
    ]
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

    best_w, best_h = 640, 480
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if aw >= w and ah >= h:
            best_w, best_h = aw, ah
            break
        if aw * ah > best_w * best_h:
            best_w, best_h = aw, ah

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── Guide box geometry ────────────────────────────────────────────────

def guide_rect(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) of the centred portrait guide rectangle."""
    gh = int(frame_h * GUIDE_H_FRAC)
    gw = int(gh * GUIDE_ASPECT)
    # Clamp width
    if gw > int(frame_w * 0.60):
        gw = int(frame_w * 0.60)
        gh = int(gw / GUIDE_ASPECT)
    cx, cy = frame_w // 2, frame_h // 2
    x1 = cx - gw // 2
    y1 = cy - gh // 2
    return x1, y1, x1 + gw, y1 + gh

# ── OCR ───────────────────────────────────────────────────────────────

def ocr_crop(crop: np.ndarray) -> list[dict]:
    """Run Tesseract on a cropped region.  Returns word-level boxes
    with coordinates relative to the crop.

    Uses digits-focused config for speed and accuracy.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Digits-only pass first (fast); fall back to general only if needed
    configs = [
        "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789",
        "--psm 11 --oem 3",
    ]

    all_words: list[dict] = []
    seen: set[str] = set()

    for cfg in configs:
        try:
            data = pytesseract.image_to_data(
                enhanced,
                lang="eng",
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue

        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            if not txt:
                continue
            conf = int(data["conf"][i])
            if conf < 15:
                continue
            key = (txt, data["left"][i], data["top"][i])
            if key in seen:
                continue
            seen.add(key)
            all_words.append({
                "text": txt,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "conf": conf,
            })

        # Early exit: if the digits-only pass already found a 202… code,
        # skip the slower general pass entirely
        code = extract_code(all_words)
        if code and code["value"].startswith("202"):
            break

    return all_words

# ── Field extraction (student code only) ──────────────────────────────

def extract_code(words: list[dict]) -> dict | None:
    """Find the 9-digit student code from OCR words.

    Priority:
      1. Any 9-digit number starting with '202' → definite match
      2. Embedded 202XXXXXX inside a longer digit string
      3. Any other 9-digit number → fallback

    Merges adjacent digit-only words on the same line.
    Returns {value, boxes} or None.
    """
    digit_groups: list[tuple[str, list[dict]]] = []
    for w in words:
        cleaned = w["text"].replace(" ", "").replace("-", "").replace(".", "")
        if not cleaned.isdigit():
            # Fix common OCR char substitutions
            cleaned2 = w["text"].replace("O", "0").replace("o", "0") \
                                .replace("l", "1").replace("I", "1") \
                                .replace(" ", "").replace("-", "").replace(".", "")
            if cleaned2.isdigit():
                cleaned = cleaned2
            else:
                continue

        merged = False
        if digit_groups:
            prev_num, prev_boxes = digit_groups[-1]
            last = prev_boxes[-1]
            if (abs(w["y"] - last["y"]) < max(w["h"], last["h"]) * 0.6
                    and w["x"] - (last["x"] + last["w"]) < last["h"] * 2):
                digit_groups[-1] = (prev_num + cleaned, prev_boxes + [w])
                merged = True
        if not merged:
            digit_groups.append((cleaned, [w]))

    best: dict | None = None
    best_priority = -1

    for num_str, boxes in digit_groups:
        # Exact 9-digit match starting with 202 → best possible
        if len(num_str) == 9 and num_str.startswith("202"):
            return {"value": num_str, "boxes": boxes}

        # Embedded 202XXXXXX in a longer string
        if len(num_str) > 9:
            m = re.search(r"(202\d{6})", num_str)
            if m and best_priority < 2:
                best = {"value": m.group(1), "boxes": boxes}
                best_priority = 2

        # Any 9-digit fallback
        if len(num_str) == 9 and best_priority < 1:
            best = {"value": num_str, "boxes": boxes}
            best_priority = 1

    return best


# ── Drawing ───────────────────────────────────────────────────────────

def draw_guide(frame: np.ndarray, gx1: int, gy1: int,
               gx2: int, gy2: int, found: bool) -> np.ndarray:
    """Draw the centred guide rectangle and dim the area outside it."""
    h, w = frame.shape[:2]

    # Fast dimming: darken whole frame, then paste bright crop back
    display = (frame * 0.3).astype(np.uint8)
    display[gy1:gy2, gx1:gx2] = frame[gy1:gy2, gx1:gx2]

    # Guide rectangle
    colour = GREEN if found else CYAN
    cv2.rectangle(display, (gx1, gy1), (gx2, gy2), colour, 3)

    # Corner accents
    corner_len = 30
    t = 5
    for (cx, cy), (dx, dy) in [
        ((gx1, gy1), (1, 1)), ((gx2, gy1), (-1, 1)),
        ((gx1, gy2), (1, -1)), ((gx2, gy2), (-1, -1)),
    ]:
        cv2.line(display, (cx, cy), (cx + dx * corner_len, cy), colour, t)
        cv2.line(display, (cx, cy), (cx, cy + dy * corner_len), colour, t)

    # Label
    label = "Centre the ID card here"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    lx = gx1 + (gx2 - gx1 - tw) // 2
    ly = gy1 - 14
    if ly < 30:
        ly = gy2 + 30
    cv2.putText(display, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    return display


def draw_code_box(frame: np.ndarray, code: dict,
                  offset_x: int, offset_y: int) -> np.ndarray:
    """Highlight the detected student code boxes (shifted to frame coords)."""
    overlay = frame.copy()
    boxes = code["boxes"]

    for b in boxes:
        pad = 6
        x1 = max(b["x"] + offset_x - pad, 0)
        y1 = max(b["y"] + offset_y - pad, 0)
        x2 = b["x"] + offset_x + b["w"] + pad
        y2 = b["y"] + offset_y + b["h"] + pad
        cv2.rectangle(overlay, (x1, y1), (x2, y2), GREEN, 3)

    first = boxes[0]
    label = f'CODIGO: {code["value"]}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    lx = first["x"] + offset_x
    ly = max(first["y"] + offset_y - 16, th + 6)
    cv2.rectangle(overlay, (lx - 4, ly - th - 6), (lx + tw + 8, ly + 6), BLACK, -1)
    cv2.putText(overlay, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    return overlay


def draw_debug_words(frame: np.ndarray, words: list[dict],
                     offset_x: int, offset_y: int) -> np.ndarray:
    """Debug: thin grey boxes + text around every OCR word."""
    for w in words:
        x = w["x"] + offset_x
        y = w["y"] + offset_y
        cv2.rectangle(frame, (x, y), (x + w["w"], y + w["h"]), GREY, 1)
        cv2.putText(frame, w["text"], (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREY, 1)
    return frame


def draw_hud(frame: np.ndarray, code: dict | None, status: str,
             fps: float, ocr_fps: float) -> np.ndarray:
    """HUD: top bar, bottom status, code display."""
    h, w = frame.shape[:2]
    display = frame.copy()

    # Top bar
    cv2.rectangle(display, (0, 0), (w, 34), DARK_BG, -1)
    cv2.putText(display, f"FPS: {fps:.0f}  |  OCR: {ocr_fps:.1f}/s",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    if code:
        val = f"Codigo: {code['value']}"
        (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(display, val, (w - tw - 16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
    else:
        cv2.putText(display, "Codigo: ---", (w - 220, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, DIM_GREY, 1)

    # Bottom bar
    cv2.rectangle(display, (0, h - 38), (w, h), DARK_BG, -1)
    cv2.putText(display, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    return display


# ── Background OCR thread ─────────────────────────────────────────────

class OCRScanner:
    """Runs Tesseract on cropped frames in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._crop: np.ndarray | None = None
        self._words: list[dict] = []
        self._code: dict | None = None
        self._running = True
        self._ocr_fps = 0.0
        self._force = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, crop: np.ndarray) -> None:
        with self._lock:
            self._crop = crop

    def force_scan(self):
        self._force.set()

    @property
    def words(self) -> list[dict]:
        with self._lock:
            return list(self._words)

    @property
    def code(self) -> dict | None:
        with self._lock:
            return self._code

    @property
    def ocr_fps(self) -> float:
        with self._lock:
            return self._ocr_fps

    def stop(self):
        self._running = False
        self._force.set()

    def _loop(self):
        count = 0
        timer = time.time()

        while self._running:
            self._force.wait(timeout=0.01)
            self._force.clear()

            with self._lock:
                crop = self._crop
                self._crop = None

            if crop is None:
                continue

            words = ocr_crop(crop)
            code = extract_code(words)

            with self._lock:
                self._words = words
                self._code = code

            count += 1
            elapsed = time.time() - timer
            if elapsed >= 1.0:
                with self._lock:
                    self._ocr_fps = count / elapsed
                count = 0
                timer = time.time()

            if DEBUG and words:
                print("  OCR:", " | ".join(
                    f'{w["text"]}({w["conf"]}%)' for w in words))


# ── Process single image ──────────────────────────────────────────────

def process_image(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not read '{path}'")
        sys.exit(1)

    print(f"Processing: {path}")
    words = ocr_crop(img)
    code = extract_code(words)

    val = code["value"] if code else "(no detectado)"
    print(f"\n  Codigo: {val}\n")

    display = img.copy()
    if DEBUG:
        draw_debug_words(display, words, 0, 0)
    if code:
        display = draw_code_box(display, code, 0, 0)

    cv2.namedWindow("OCR Result", cv2.WINDOW_NORMAL)
    h, w = display.shape[:2]
    scale = DISPLAY_WIDTH / w
    cv2.resizeWindow("OCR Result", DISPLAY_WIDTH, int(h * scale))
    cv2.imshow("OCR Result", display)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Live camera loop ──────────────────────────────────────────────────

def camera_loop() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        sys.exit(1)

    cam_w, cam_h = configure_camera(cap)
    disp_scale = DISPLAY_WIDTH / cam_w
    disp_w = DISPLAY_WIDTH
    disp_h = int(cam_h * disp_scale)

    # Guide box in frame coordinates
    gx1, gy1, gx2, gy2 = guide_rect(cam_w, cam_h)

    print("+" + "=" * 56 + "+")
    print("|  UNIANDES — STUDENT CODE READER                       |")
    print("|                                                        |")
    print(f"|  Camera : {cam_w}x{cam_h}".ljust(57) + "|")
    print(f"|  Guide  : {gx2-gx1}x{gy2-gy1} (scan region)".ljust(57) + "|")
    print("|                                                        |")
    print("|  Centre the ID card inside the box.                    |")
    print("|  The student code (9 digits, 202...) is detected       |")
    print("|  automatically.                                        |")
    print("|                                                        |")
    print("|  SPACE -> force scan   C -> copy   S -> save           |")
    print("|  D -> debug words      ESC / Q -> quit                 |")
    print("+" + "=" * 56 + "+\n")

    win_name = "Student Code Reader"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)

    scanner = OCRScanner()
    status = "Centre the ID card inside the box"
    show_debug = DEBUG
    last_code_value = ""

    fps = 0.0
    frame_count = 0
    fps_timer = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            break

        # Crop from UNMIRRORED frame for OCR (text reads correctly)
        crop = raw_frame[gy1:gy2, gx1:gx2]
        scanner.submit(crop)

        # Mirror the frame for display (easier to position the card)
        frame = cv2.flip(raw_frame, 1)

        # Get results
        code = scanner.code
        words = scanner.words
        ocr_fps = scanner.ocr_fps

        # Guide box coords are centred → same after mirror
        display = draw_guide(frame, gx1, gy1, gx2, gy2, found=code is not None)

        # OCR coords need x-mirroring within the crop region
        crop_w = gx2 - gx1
        if show_debug:
            mirrored_words = [
                {**w, "x": crop_w - w["x"] - w["w"]} for w in words
            ]
            display = draw_debug_words(display, mirrored_words, gx1, gy1)

        if code:
            mirrored_code = {
                "value": code["value"],
                "boxes": [{**b, "x": crop_w - b["x"] - b["w"]} for b in code["boxes"]],
            }
            display = draw_code_box(display, mirrored_code, gx1, gy1)
            last_code_value = code["value"]
            status = f"Found: {code['value']}"
        else:
            status = "Scanning... centre ID card in the box"

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        display = draw_hud(display, code, status, fps, ocr_fps)
        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        elif key == ord(" "):
            scanner.force_scan()
            status = "Forced scan..."
        elif key in (ord("d"), ord("D")):
            show_debug = not show_debug
            print(f"  Debug: {'ON' if show_debug else 'OFF'}")
        elif key in (ord("s"), ord("S")):
            fname = f"ocr_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, raw_frame)      # save unmirrored
            print(f"  Saved: {fname}")
            status = f"Saved: {fname}"
        elif key in (ord("c"), ord("C")):
            if last_code_value:
                if copy_to_clipboard(last_code_value):
                    print(f"  Copied: {last_code_value}")
                    status = f"Copied: {last_code_value}"
                else:
                    status = "Clipboard failed"
            else:
                status = "Nothing to copy yet"

    scanner.stop()
    cap.release()
    cv2.destroyAllWindows()

    if last_code_value:
        print(f"\n  Last detected code: {last_code_value}\n")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        process_image(args[0])
    else:
        camera_loop()


if __name__ == "__main__":
    main()
