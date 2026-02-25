# ============================================================
# Name        : Shravan
# Roll No     : 2301010465
# Course      : Image Processing & Computer Vision
# Unit        : Image Sensing & Acquisition
# Assignment  : Smart Document Scanner & Quality Analysis System
# Date        : 24 February 2026
# ============================================================

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# TASK 1: Project Setup and Introduction
# ─────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  Smart Document Scanner & Quality Analysis System")
print("=" * 60)
print("""
This system simulates a real-world document digitization pipeline.
It demonstrates how:
  • Image acquisition converts physical documents to digital form.
  • Sampling (resolution) affects fine text details and sharpness.
  • Quantization (bit-depth) affects tonal representation and artifacts.
  • These factors jointly determine OCR accuracy and readability.
""")
print("=" * 60)

# ─────────────────────────────────────────────
# TASK 2: Image Acquisition
# ─────────────────────────────────────────────

def load_image(source: str = "webcam") -> np.ndarray:
    """Load a document image from a file path or webcam."""
    if source.lower() == "webcam":
        print("\n[INFO] Attempting to capture image from webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[WARN] Webcam not available. Generating a synthetic document image.")
            return _generate_synthetic_document()
        print("[INFO] Press SPACE to capture or ESC to cancel.")
        frame = None
        while True:
            ret, f = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam – Press SPACE to capture", f)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frame = f
                break
            elif key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame is None:
            print("[WARN] No frame captured. Generating a synthetic document image.")
            return _generate_synthetic_document()
        return frame
    else:
        if not os.path.exists(source):
            print(f"[WARN] File '{source}' not found. Generating a synthetic document image.")
            return _generate_synthetic_document()
        img = cv2.imread(source)
        if img is None:
            print("[WARN] Could not read image. Generating a synthetic document image.")
            return _generate_synthetic_document()
        print(f"[INFO] Loaded image: {source}  |  Shape: {img.shape}")
        return img


def _generate_synthetic_document() -> np.ndarray:
    """Create a realistic synthetic text document for demo purposes."""
    h, w = 512, 512
    doc = np.ones((h, w, 3), dtype=np.uint8) * 255   # white background

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        ("SMART DOCUMENT SCANNER", (30, 50), 0.7, 2),
        ("Image Processing & Computer Vision", (20, 90), 0.5, 1),
        ("-" * 55, (20, 110), 0.4, 1),
        ("Task: Analyze sampling and quantization effects", (20, 145), 0.45, 1),
        ("on document image quality and OCR accuracy.", (20, 175), 0.45, 1),
        ("Resolution levels tested: 512x512, 256x256, 128x128", (20, 215), 0.42, 1),
        ("Bit-depth levels: 8-bit (256), 4-bit (16), 2-bit (4)", (20, 245), 0.42, 1),
        ("-" * 55, (20, 265), 0.4, 1),
        ("Fine detail text: abcdefghijklmnopqrstuvwxyz 0-9", (20, 300), 0.4, 1),
        ("The quick brown fox jumps over the lazy dog.", (20, 325), 0.4, 1),
        ("Lorem ipsum dolor sit amet, consectetur.", (20, 350), 0.38, 1),
        ("Sampling reduces spatial resolution (pixels/inch).", (20, 385), 0.38, 1),
        ("Quantization reduces gray-level depth (bits/pixel).", (20, 410), 0.38, 1),
        ("Both degrade readability & OCR accuracy.", (20, 435), 0.38, 1),
        ("-" * 55, (20, 455), 0.4, 1),
        ("Roll No: ________   Date: Feb 2026", (20, 490), 0.42, 1),
    ]
    for text, org, scale, thickness in lines:
        cv2.putText(doc, text, org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    print("[INFO] Synthetic document image generated (512×512).")
    return doc


def acquire_document(source: str = "webcam") -> tuple:
    """Acquire, resize, and convert document image. Returns (original_rgb, gray)."""
    raw = load_image(source)
    resized = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (512×512, Colour)")
    axes[0].axis("off")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale Conversion")
    axes[1].axis("off")
    plt.suptitle("Task 2 – Image Acquisition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/task2_acquisition.png", dpi=150)
    plt.show()
    print("[INFO] Task 2 output saved → outputs/task2_acquisition.png")

    return resized, gray


# ─────────────────────────────────────────────
# TASK 3: Image Sampling (Resolution Analysis)
# ─────────────────────────────────────────────

def sample_image(gray: np.ndarray) -> dict:
    """Down-sample then up-scale to 512 for comparison."""
    resolutions = {"High (512×512)": 512, "Medium (256×256)": 256, "Low (128×128)": 128}
    sampled = {}

    for label, res in resolutions.items():
        # Down-sample
        small = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
        # Up-scale back to 512 for fair visual comparison
        restored = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)
        sampled[label] = (small, restored)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (label, (_, restored)) in zip(axes, sampled.items()):
        ax.imshow(restored, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.suptitle("Task 3 – Sampling Analysis (up-scaled to 512 for display)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/task3_sampling.png", dpi=150)
    plt.show()
    print("[INFO] Task 3 output saved → outputs/task3_sampling.png")

    return sampled


# ─────────────────────────────────────────────
# TASK 4: Image Quantization (Bit-Depth Analysis)
# ─────────────────────────────────────────────

def quantize_image(gray: np.ndarray) -> dict:
    """Reduce gray levels to simulate lower bit-depth."""
    bit_configs = {
        "8-bit (256 levels)": 256,
        "4-bit (16 levels)":   16,
        "2-bit (4 levels)":     4,
    }
    quantized = {}

    for label, levels in bit_configs.items():
        factor = 256 // levels
        q = (gray // factor) * factor
        quantized[label] = q.astype(np.uint8)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (label, q_img) in zip(axes, quantized.items()):
        ax.imshow(q_img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.suptitle("Task 4 – Quantization Analysis (gray-level reduction)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/task4_quantization.png", dpi=150)
    plt.show()
    print("[INFO] Task 4 output saved → outputs/task4_quantization.png")

    return quantized


# ─────────────────────────────────────────────
# TASK 5: Quality Observation & Comparison Figure
# ─────────────────────────────────────────────

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (higher = better quality)."""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def quality_analysis(gray: np.ndarray, sampled: dict, quantized: dict) -> None:
    """Print observations and display the full comparison figure."""

    print("\n" + "=" * 60)
    print("  TASK 5 – Quality Observations & Analysis")
    print("=" * 60)

    # --- Sampling ---
    print("\n📷 SAMPLING (Resolution) Analysis:")
    print("-" * 50)
    prev_psnr = None
    for label, (small, restored) in sampled.items():
        psnr = compute_psnr(gray, restored)
        psnr_str = f"{psnr:.2f} dB" if psnr != float("inf") else "∞ (lossless)"
        print(f"  {label:<22}  PSNR: {psnr_str}")

    print("""
  Observations:
  • 512×512 (High): Maximum text sharpness; fine strokes and
    serifs are clearly preserved. Ideal for OCR.
  • 256×256 (Medium): Slight blurring of small fonts; most
    body text is still legible. Acceptable for OCR with
    common fonts ≥ 10 pt.
  • 128×128 (Low): Significant loss of fine text detail;
    characters merge or lose serifs. OCR accuracy drops
    markedly, especially for small fonts or stylised scripts.
""")

    # --- Quantization  ---
    print("🎨 QUANTIZATION (Bit-Depth) Analysis:")
    print("-" * 50)
    for label, q_img in quantized.items():
        psnr = compute_psnr(gray, q_img)
        psnr_str = f"{psnr:.2f} dB" if psnr != float("inf") else "∞ (lossless)"
        print(f"  {label:<25}  PSNR: {psnr_str}")

    print("""
  Observations:
  • 8-bit (256 levels): Full tonal range; no visible banding.
    Readability and OCR suitability: Excellent.
  • 4-bit (16 levels): Mild false-contouring (banding) in
    smooth gradient regions. Text body still readable but
    fine anti-aliased edges may suffer.
  • 2-bit (4 levels): Heavy posterisation; large flat regions
    and severe loss of mid-tones. Severely hampers OCR
    accuracy for anything other than high-contrast bold text.
""")

    print("  OCR Suitability Summary:")
    print("  ┌───────────────────────────┬──────────────────────┐")
    print("  │ Configuration             │ OCR Suitability      │")
    print("  ├───────────────────────────┼──────────────────────┤")
    print("  │ 512×512 + 8-bit           │ Excellent            │")
    print("  │ 256×256 + 8-bit           │ Good                 │")
    print("  │ 128×128 + 8-bit           │ Poor                 │")
    print("  │ 512×512 + 4-bit           │ Good                 │")
    print("  │ 512×512 + 2-bit           │ Very Poor            │")
    print("  └───────────────────────────┴──────────────────────┘")
    print("=" * 60)

    # --- Combined comparison figure ---
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Smart Document Scanner – Full Quality Comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 7, figure=fig, wspace=0.3, hspace=0.4)

    # Row labels as axes
    sample_labels = list(sampled.keys())
    quant_labels  = list(quantized.keys())

    # Original
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(gray, cmap="gray")
    ax_orig.set_title("Original\n(512×512\n8-bit)", fontsize=9, fontweight="bold")
    ax_orig.axis("off")

    # Divider text
    ax_div1 = fig.add_subplot(gs[:, 1])
    ax_div1.text(0.5, 0.5, "SAMPLING\nANALYSIS", ha="center", va="center",
                 fontsize=10, fontweight="bold", rotation=90,
                 transform=ax_div1.transAxes, color="#2255aa")
    ax_div1.axis("off")

    # Sampled images (columns 2-4)
    for col, (label, (_, restored)) in enumerate(sampled.items(), start=2):
        ax = fig.add_subplot(gs[:, col])
        ax.imshow(restored, cmap="gray")
        psnr = compute_psnr(gray, restored)
        psnr_str = f"{psnr:.1f} dB" if psnr != float("inf") else "∞"
        ax.set_title(f"{label}\nPSNR: {psnr_str}", fontsize=8)
        ax.axis("off")

    # Divider text
    ax_div2 = fig.add_subplot(gs[:, 5])
    ax_div2.text(0.5, 0.5, "QUANTIZATION\nANALYSIS", ha="center", va="center",
                 fontsize=10, fontweight="bold", rotation=90,
                 transform=ax_div2.transAxes, color="#aa2222")
    ax_div2.axis("off")

    # Quantized images
    for row, (label, q_img) in enumerate(quantized.items()):
        ax = fig.add_subplot(gs[row if row < 2 else 1, 6])
        ax.imshow(q_img, cmap="gray")
        psnr = compute_psnr(gray, q_img)
        psnr_str = f"{psnr:.1f} dB" if psnr != float("inf") else "∞"
        ax.set_title(f"{label}\nPSNR: {psnr_str}", fontsize=8)
        ax.axis("off")

    plt.savefig("outputs/task5_full_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Task 5 full comparison saved → outputs/task5_full_comparison.png")


# ─────────────────────────────────────────────
# MAIN – Run all tasks
# ─────────────────────────────────────────────

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "synthetic"

    print(f"\n[INFO] Image source: '{source}'")

    # Task 2
    original_bgr, gray = acquire_document(source)

    # Task 3
    sampled = sample_image(gray)

    # Task 4
    quantized = quantize_image(gray)

    # Task 5
    quality_analysis(gray, sampled, quantized)

    print("\n[DONE] All tasks completed. Check the outputs/ folder for saved figures.")


if __name__ == "__main__":
    main()