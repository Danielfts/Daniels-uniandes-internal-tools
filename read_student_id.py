#!/usr/bin/env python3
"""
Uniandes Student ID Card Reader
================================
Opens the camera and scans student ID cards in real time.
Press SPACE to capture and process a frame, ESC/Q to quit.

Usage:
    python read_student_id.py              # live camera mode
    python read_student_id.py <image_path> # process a single image
    python read_student_id.py --debug      # show raw OCR output
"""

import sys
import os
import re
import time
import cv2
import numpy as np
import pytesseract

try:
    from pyzbar.pyzbar import decode as decode_barcodes
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("⚠  pyzbar no disponible (instalar zbar: sudo dnf install -y zbar)")
    print("   Continuando sin lectura de códigos de barras...\n")

DEBUG = "--debug" in sys.argv

# ── OCR & Barcode helpers ─────────────────────────────────────────────

def extract_text(img: np.ndarray) -> str:
    """Run OCR on multiple preprocessed variants of the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    variants = [
        gray,
        clahe.apply(gray),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            clahe.apply(gray), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        ),
    ]

    all_text: list[str] = []
    for v in variants:
        for lang in ("spa", "eng"):
            try:
                text = pytesseract.image_to_string(v, lang=lang, config="--psm 6")
                all_text.append(text)
            except Exception:
                pass
    return "\n".join(all_text)


def read_barcodes(img: np.ndarray) -> list[dict]:
    """Detect and decode barcodes / QR codes with multiple preprocessing attempts."""
    if not PYZBAR_AVAILABLE:
        return []
    results: list[dict] = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Upscale small images for better barcode detection
    h, w = gray.shape[:2]
    if w < 1000:
        scale = 1000 / w
        gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        gray_up = gray

    # Multiple preprocessing variants to maximise barcode detection
    variants = [
        img,                                                              # original color
        gray,                                                             # grayscale
        gray_up,                                                          # upscaled grayscale
        clahe.apply(gray),                                                # CLAHE enhanced
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu
        cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1],              # fixed threshold
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 10),                 # adaptive
    ]

    # Also try sharpened version
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(sharpened)

    for frame in variants:
        try:
            for bc in decode_barcodes(frame):
                entry = {"type": bc.type, "data": bc.data.decode("utf-8", errors="replace")}
                if entry not in results:
                    results.append(entry)
        except Exception:
            pass

    return results


# ── Parsing ───────────────────────────────────────────────────────────

def parse_student_info(raw_text: str, barcodes: list[dict]) -> dict:
    """Extract structured fields from raw OCR text."""
    info: dict = {
        "nombre": None,
        "programa": None,
        "codigo_estudiante": None,
        "documento": None,
        "codigos_barras": barcodes,
    }

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # ── Numbers (student code & document) ──
    number_candidates: list[str] = []
    for line in lines:
        number_candidates.extend(re.findall(r"\b\d{7,12}\b", line))
    seen: set[str] = set()
    unique_nums: list[str] = []
    for n in number_candidates:
        if n not in seen:
            seen.add(n)
            unique_nums.append(n)

    for n in unique_nums:
        if re.match(r"^20[0-9]{2}[0-9]{4,6}$", n):
            info["codigo_estudiante"] = n
            break

    for n in unique_nums:
        if n != info["codigo_estudiante"] and len(n) >= 8:
            info["documento"] = n
            break

    # ── Program ──
    prog_re = re.compile(
        r"(Ingenier[ií]a\s+\w+|Administraci[oó]n\s+\w+|"
        r"Ciencia\s+\w+|Derecho|Econom[ií]a|Medicina|"
        r"Arquitectura|Dise[ñn]o\s+\w+|Matem[aá]ticas|"
        r"F[ií]sica|Qu[ií]mica|Biolog[ií]a|"
        r"Licenciatura\s+[\w\s]+|Maestr[ií]a\s+[\w\s]+|"
        r"Literatura|Filosof[ií]a|M[uú]sica|Arte|"
        r"Psicolog[ií]a|Antropolog[ií]a|"
        r"Ciencia\s+Pol[ií]tica|Gobierno)",
        re.IGNORECASE,
    )
    for line in lines:
        m = prog_re.search(line)
        if m:
            info["programa"] = m.group(0).strip()
            break

    # ── Name ──
    name_re = re.compile(
        r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,5}$"
    )
    skip_kw = [
        "ingenier", "carrera", "bogot", "colombia", "universidad",
        "administra", "ciencia", "derecho", "econom", "licenci",
        "maestr", "medicin", "arquitec",
    ]
    name_candidates = [
        l for l in lines
        if name_re.match(l) and not any(kw in l.lower() for kw in skip_kw)
    ]
    if name_candidates:
        info["nombre"] = max(name_candidates, key=len)

    return info


# ── Display ───────────────────────────────────────────────────────────

def format_info(info: dict) -> list[str]:
    """Return a list of display lines for the parsed info."""
    labels = {
        "nombre": "Nombre",
        "programa": "Programa",
        "codigo_estudiante": "Codigo",
        "documento": "Documento",
    }
    out = []
    for key, label in labels.items():
        value = info.get(key) or "(no detectado)"
        out.append(f"{label}: {value}")
    for i, bc in enumerate(info.get("codigos_barras", []), 1):
        out.append(f"Barcode [{i}]: {bc['type']} -> {bc['data']}")
    return out


def print_info(info: dict) -> None:
    """Pretty-print info to the terminal."""
    print("\n" + "=" * 50)
    print("  UNIANDES STUDENT ID - EXTRACTED INFO")
    print("=" * 50)
    for line in format_info(info):
        print(f"  {line}")
    print("=" * 50 + "\n")


def overlay_info(frame: np.ndarray, info: dict) -> np.ndarray:
    """Draw extracted info as an overlay on the camera frame."""
    overlay = frame.copy()
    lines = format_info(info)
    y0 = 30
    for i, line in enumerate(lines):
        y = y0 + i * 28
        # Shadow
        cv2.putText(overlay, line, (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # Text
        color = (0, 255, 0) if "(no detectado)" not in line else (0, 0, 255)
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return overlay


# ── Process a single image ────────────────────────────────────────────

def process_image(path: str) -> None:
    """Load an image file, run OCR + barcodes, print results."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: no se pudo leer '{path}'")
        sys.exit(1)
    print(f"Procesando: {path}")
    print("  [1/3] Codigos de barras...")
    barcodes = read_barcodes(img)
    print("  [2/3] OCR...")
    raw_text = extract_text(img)
    print("  [3/3] Interpretando...")
    info = parse_student_info(raw_text, barcodes)
    print_info(info)
    if DEBUG:
        print("\n--- RAW OCR ---\n" + raw_text + "\n--- END ---\n")


# ── Live camera mode ──────────────────────────────────────────────────

def camera_loop() -> None:
    """Open the camera and scan student IDs in real time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("+" + "=" * 50 + "+")
    print("|  UNIANDES STUDENT ID SCANNER - MODO CAMARA      |")
    print("|                                                  |")
    print("|  ESPACIO  -> capturar y procesar frame           |")
    print("|  S        -> guardar captura como imagen         |")
    print("|  ESC / Q  -> salir                               |")
    print("+" + "=" * 50 + "+\n")

    last_info: dict | None = None
    processing = False
    status_msg = "Apunta la camara al carnet y presiona ESPACIO"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: no se pudo leer frame de la camara.")
            break

        display = frame.copy()

        # Draw guide rectangle in the center
        h, w = frame.shape[:2]
        margin_x, margin_y = int(w * 0.15), int(h * 0.1)
        cv2.rectangle(
            display,
            (margin_x, margin_y),
            (w - margin_x, h - margin_y),
            (0, 255, 255), 2,
        )

        # Status bar at the bottom
        cv2.rectangle(display, (0, h - 40), (w, h), (40, 40, 40), -1)
        cv2.putText(
            display, status_msg,
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        # If we have results, overlay them
        if last_info is not None:
            display = overlay_info(display, last_info)

        cv2.imshow("Uniandes ID Scanner", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):  # ESC or Q
            break

        elif key == ord(" ") and not processing:
            processing = True
            status_msg = "Procesando... espera"
            # Redraw with status
            cv2.rectangle(display, (0, h - 40), (w, h), (40, 40, 40), -1)
            cv2.putText(
                display, status_msg,
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
            )
            cv2.imshow("Uniandes ID Scanner", display)
            cv2.waitKey(1)

            # Crop to guide area for better OCR
            crop = frame[margin_y : h - margin_y, margin_x : w - margin_x]

            print("\n  Captura tomada — procesando...")
            # Try barcodes on BOTH the crop and the full frame
            barcodes = read_barcodes(crop)
            if not barcodes:
                barcodes = read_barcodes(frame)
            raw_text = extract_text(crop)
            last_info = parse_student_info(raw_text, barcodes)

            print_info(last_info)
            if DEBUG:
                print("--- RAW OCR ---\n" + raw_text + "\n--- END ---\n")

            status_msg = "Listo! ESPACIO=escanear | S=guardar | ESC=salir"
            processing = False

        elif key in (ord("s"), ord("S")):
            fname = f"carnet_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  Imagen guardada: {fname}")
            status_msg = f"Guardado: {fname}"

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────

def main():
    # Filter out flags
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        # Image file mode
        process_image(args[0])
    else:
        # Live camera mode
        camera_loop()


if __name__ == "__main__":
    main()
