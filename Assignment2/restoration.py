# ============================================================
# Name        : Shravan
# Roll No     : 2301010465
# Course      : Image Processing & Computer Vision
# Unit        : Image Restoration & Noise Modeling
# Assignment  : Image Restoration for Surveillance Camera Systems
# Date        : 24 February 2026
# ============================================================

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# TASK 1: Image Selection and Preprocessing
# ─────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  Image Restoration for Surveillance Camera Systems")
print("=" * 65)
print("""
This system simulates real-world surveillance image degradation
and restores quality using classical spatial filtering techniques.

Pipeline:
  1. Load surveillance-style image → grayscale
  2. Add Gaussian noise  (simulates sensor/thermal noise)
  3. Add Salt-and-Pepper noise (simulates transmission errors)
  4. Apply Mean / Median / Gaussian filters for restoration
  5. Evaluate using MSE and PSNR; select best filter per noise type
""")
print("=" * 65)


def _generate_synthetic_surveillance() -> np.ndarray:
    """Generate a synthetic surveillance-style scene (street / corridor)."""
    h, w = 512, 512
    img = np.ones((h, w, 3), dtype=np.uint8) * 30  # dark background

    # Sky / ceiling gradient
    for y in range(120):
        v = int(30 + y * 0.6)
        img[y, :] = [v, v, v]

    # Road / floor
    img[320:, :] = 55

    # Buildings / walls (rectangles)
    cv2.rectangle(img, (20, 80),  (180, 320), (90, 90, 90), -1)
    cv2.rectangle(img, (200, 120), (360, 320), (75, 75, 75), -1)
    cv2.rectangle(img, (380, 100), (490, 320), (85, 85, 85), -1)

    # Windows
    for bx, by, bw, bh in [(20, 80, 160, 240), (200, 120, 160, 200), (380, 100, 110, 220)]:
        for r in range(by + 20, by + bh - 20, 45):
            for c in range(bx + 15, bx + bw - 15, 40):
                brightness = np.random.choice([200, 220, 60])
                cv2.rectangle(img, (c, r), (c + 25, r + 30),
                              (brightness, brightness, int(brightness * 0.85)), -1)

    # Car silhouette
    cv2.rectangle(img, (80, 340),  (230, 390), (110, 110, 110), -1)
    cv2.rectangle(img, (110, 310), (200, 342), (100, 100, 100), -1)
    cv2.ellipse(img, (110, 393), (22, 22), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (210, 393), (22, 22), 0, 0, 360, (50, 50, 50), -1)

    # Person silhouette
    cv2.rectangle(img, (300, 330), (330, 410), (60, 60, 60), -1)
    cv2.circle(img, (315, 318), 18, (60, 60, 60), -1)

    # Street lamp
    cv2.line(img, (440, 180), (440, 325), (120, 120, 120), 4)
    cv2.ellipse(img, (440, 175), (30, 10), 0, 0, 180, (200, 200, 160), -1)

    # Overlay text for realism
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "CAM-04  CH-1", (5, 20),  font, 0.45, (180, 180, 180), 1)
    cv2.putText(img, "2026-02-14 23:47:09", (5, 505 if h > 500 else h - 10),
                font, 0.4, (180, 180, 180), 1)

    print("[INFO] Synthetic surveillance image generated (512×512).")
    return img


def load_image(source: str = "synthetic") -> np.ndarray:
    """Load image from file path, webcam, or generate synthetic."""
    if source.lower() == "synthetic":
        return _generate_synthetic_surveillance()
    if source.lower() == "webcam":
        print("[INFO] Trying webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[WARN] Webcam unavailable. Using synthetic image.")
            return _generate_synthetic_surveillance()
        print("[INFO] Press SPACE to capture or ESC to cancel.")
        frame = None
        while True:
            ret, f = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam – SPACE to capture", f)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frame = f
                break
            elif key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return frame if frame is not None else _generate_synthetic_surveillance()
    if not os.path.exists(source):
        print(f"[WARN] '{source}' not found. Using synthetic image.")
        return _generate_synthetic_surveillance()
    img = cv2.imread(source)
    if img is None:
        print("[WARN] Could not read image. Using synthetic image.")
        return _generate_synthetic_surveillance()
    print(f"[INFO] Loaded: {source}  |  Shape: {img.shape}")
    return img


def preprocess(source: str = "synthetic") -> np.ndarray:
    """Load, resize to 512×512, convert to grayscale, display, and save."""
    raw = load_image(source)
    resized = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Surveillance Image (Colour)")
    axes[0].axis("off")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale Conversion")
    axes[1].axis("off")
    plt.suptitle("Task 1 – Image Acquisition & Preprocessing", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/task1_original.png", dpi=150)
    plt.show()

    cv2.imwrite("outputs/original_gray.png", gray)
    print("[INFO] Task 1 outputs saved → outputs/task1_original.png, outputs/original_gray.png")
    return gray


# ─────────────────────────────────────────────
# TASK 2: Noise Modeling
# ─────────────────────────────────────────────

def add_gaussian_noise(gray: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Add Gaussian (sensor) noise to a grayscale image."""
    noise = np.random.normal(mean, sigma, gray.shape)
    noisy = np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(gray: np.ndarray, salt_prob: float = 0.02,
                          pepper_prob: float = 0.02) -> np.ndarray:
    """Add Salt-and-Pepper (transmission error) noise."""
    noisy = gray.copy()
    total = gray.size
    # Salt
    num_salt = int(total * salt_prob)
    coords = [np.random.randint(0, d, num_salt) for d in gray.shape]
    noisy[tuple(coords)] = 255
    # Pepper
    num_pepper = int(total * pepper_prob)
    coords = [np.random.randint(0, d, num_pepper) for d in gray.shape]
    noisy[tuple(coords)] = 0
    return noisy


def model_noise(gray: np.ndarray) -> dict:
    """Generate both noise types, display, and save."""
    np.random.seed(42)   # reproducibility
    gaussian_noisy  = add_gaussian_noise(gray, sigma=25)
    sp_noisy        = add_salt_pepper_noise(gray, salt_prob=0.02, pepper_prob=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, img, title in zip(axes,
                               [gray, gaussian_noisy, sp_noisy],
                               ["Original (Clean)",
                                "Gaussian Noise\n(σ=25, sensor noise)",
                                "Salt-and-Pepper\n(p=2%+2%, transmission error)"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle("Task 2 – Noise Modeling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/task2_noisy_images.png", dpi=150)
    plt.show()

    cv2.imwrite("outputs/noisy_gaussian.png",  gaussian_noisy)
    cv2.imwrite("outputs/noisy_saltpepper.png", sp_noisy)
    print("[INFO] Task 2 outputs saved → outputs/task2_noisy_images.png")

    return {"gaussian": gaussian_noisy, "salt_pepper": sp_noisy}


# ─────────────────────────────────────────────
# TASK 3: Image Restoration Techniques
# ─────────────────────────────────────────────

def apply_filters(noisy: np.ndarray, noise_label: str) -> dict:
    """Apply Mean, Median, and Gaussian filters; display and save results."""
    kernel = 5   # kernel size for all filters

    mean_restored     = cv2.blur(noisy, (kernel, kernel))
    median_restored   = cv2.medianBlur(noisy, kernel)
    gaussian_restored = cv2.GaussianBlur(noisy, (kernel, kernel), sigmaX=1.5)

    restored = {
        "Mean Filter":     mean_restored,
        "Median Filter":   median_restored,
        "Gaussian Filter": gaussian_restored,
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, img) in zip(axes,
                                  [("Noisy Input\n(" + noise_label + ")", noisy)] +
                                  list(restored.items())):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle(f"Task 3 – Restoration Filters  [{noise_label}]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = noise_label.lower().replace(" ", "_").replace("-", "")
    plt.savefig(f"outputs/task3_restoration_{fname}.png", dpi=150)
    plt.show()

    for filter_name, img in restored.items():
        safe = filter_name.lower().replace(" ", "_")
        cv2.imwrite(f"outputs/restored_{fname}_{safe}.png", img)
    print(f"[INFO] Task 3 outputs saved for [{noise_label}]")

    return restored


# ─────────────────────────────────────────────
# TASK 4: Performance Evaluation
# ─────────────────────────────────────────────

def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    return float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    mse = compute_mse(original, processed)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def evaluate(gray: np.ndarray, noisy_dict: dict,
             restored_gaussian: dict, restored_sp: dict) -> dict:
    """Compute MSE and PSNR for all images; return results dict."""
    results = {}

    # Noisy baselines
    for label, noisy in noisy_dict.items():
        tag = "Gaussian Noisy" if label == "gaussian" else "S&P Noisy"
        results[tag] = {
            "image": noisy,
            "mse":  compute_mse(gray, noisy),
            "psnr": compute_psnr(gray, noisy),
        }

    # Restored
    for filter_name, img in restored_gaussian.items():
        results[f"[Gaussian] {filter_name}"] = {
            "image": img,
            "mse":  compute_mse(gray, img),
            "psnr": compute_psnr(gray, img),
        }
    for filter_name, img in restored_sp.items():
        results[f"[S&P] {filter_name}"] = {
            "image": img,
            "mse":  compute_mse(gray, img),
            "psnr": compute_psnr(gray, img),
        }

    return results


# ─────────────────────────────────────────────
# TASK 5: Analytical Discussion
# ─────────────────────────────────────────────

def analytical_discussion(gray: np.ndarray, noisy_dict: dict,
                           restored_gaussian: dict, restored_sp: dict) -> None:
    """Print filter-wise performance table, identify best filters, and show comparison figure."""

    results = evaluate(gray, noisy_dict, restored_gaussian, restored_sp)

    print("\n" + "=" * 65)
    print("  TASK 5 – Performance Evaluation & Analytical Discussion")
    print("=" * 65)

    # ── Metrics Table ──────────────────────────────────────────
    print(f"\n{'Variant':<35} {'MSE':>10} {'PSNR (dB)':>12}")
    print("-" * 60)
    for label, data in results.items():
        psnr_str = f"{data['psnr']:.2f}" if data['psnr'] != float("inf") else "∞"
        print(f"  {label:<33} {data['mse']:>10.2f} {psnr_str:>12}")
    print("-" * 60)

    # ── Best filter identification ─────────────────────────────
    gauss_filters = {k: v for k, v in results.items() if k.startswith("[Gaussian]")}
    sp_filters    = {k: v for k, v in results.items() if k.startswith("[S&P]")}

    best_gauss = max(gauss_filters, key=lambda k: gauss_filters[k]["psnr"])
    best_sp    = max(sp_filters,    key=lambda k: sp_filters[k]["psnr"])

    print(f"\n  ✅ Best filter for Gaussian noise  : {best_gauss}  "
          f"(PSNR = {gauss_filters[best_gauss]['psnr']:.2f} dB)")
    print(f"  ✅ Best filter for S&P noise       : {best_sp}  "
          f"(PSNR = {sp_filters[best_sp]['psnr']:.2f} dB)")

    # ── Theoretical justification ──────────────────────────────
    print("""
  THEORETICAL JUSTIFICATION
  ──────────────────────────
  Gaussian Noise (sensor/thermal):
    • Gaussian noise adds smooth, continuous perturbations to every
      pixel.  Averaging filters (Mean, Gaussian) are statistically
      optimal because they reduce variance while preserving overall
      structure.
    • Gaussian filter wins over plain mean filter because its
      weighted kernel down-weights distant pixels, preserving edges
      slightly better while still smoothing noise effectively.
    • Median filter is less effective here because Gaussian noise
      does NOT introduce extreme outliers — the median stays close
      to the noisy value rather than the true pixel value.

  Salt-and-Pepper Noise (transmission errors):
    • S&P noise introduces extreme impulse values (0 or 255) at
      random pixel locations.
    • Median filter excels because it replaces each pixel with the
      sorted middle value of its neighbourhood — which is always an
      actual image value, not a corrupted spike.
    • Mean and Gaussian filters spread the spike energy across
      neighbouring pixels, leaving visible blurring artefacts and
      residual bright/dark spots.

  General Observation:
    • Higher PSNR (lower MSE) ↔ better restoration quality.
    • There is no single "best" filter: noise type dictates choice.
    • In real surveillance pipelines, adaptive filters (e.g.,
      Wiener, bilateral, NLM) combine the advantages of both.
""")
    print("=" * 65)

    # ── Combined Comparison Figure ─────────────────────────────
    rows_data = [
        ("Clean Original",      gray),
        ("Gaussian Noisy",      noisy_dict["gaussian"]),
        ("[G] Mean Restored",   restored_gaussian["Mean Filter"]),
        ("[G] Median Restored", restored_gaussian["Median Filter"]),
        ("[G] Gaussian Restored", restored_gaussian["Gaussian Filter"]),
        ("S&P Noisy",           noisy_dict["salt_pepper"]),
        ("[S] Mean Restored",   restored_sp["Mean Filter"]),
        ("[S] Median Restored", restored_sp["Median Filter"]),
        ("[S] Gaussian Restored", restored_sp["Gaussian Filter"]),
    ]

    cols = 3
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 13))
    fig.suptitle("Smart Surveillance Image Restoration – Full Comparison",
                 fontsize=13, fontweight="bold")

    layout = [
        ("Clean Original",       gray),
        ("Gaussian Noisy",       noisy_dict["gaussian"]),
        ("S&P Noisy",            noisy_dict["salt_pepper"]),
        ("[G] Mean Filter",      restored_gaussian["Mean Filter"]),
        ("[G] Median Filter",    restored_gaussian["Median Filter"]),
        ("[G] Gaussian Filter",  restored_gaussian["Gaussian Filter"]),
        ("[S&P] Mean Filter",    restored_sp["Mean Filter"]),
        ("[S&P] Median Filter",  restored_sp["Median Filter"]),
        ("[S&P] Gaussian Filter",restored_sp["Gaussian Filter"]),
    ]

    for idx, (ax, (title, img)) in enumerate(zip(axes.flat, layout)):
        ax.imshow(img, cmap="gray")
        key = None
        for k, v in results.items():
            if np.array_equal(v["image"], img):
                key = k
                break
        if key:
            psnr_str = f"{results[key]['psnr']:.1f} dB" if results[key]["psnr"] != float("inf") else "∞"
            subtitle = f"PSNR: {psnr_str}  MSE: {results[key]['mse']:.1f}"
        else:
            subtitle = "Reference"
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/task5_full_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Task 5 full comparison saved → outputs/task5_full_comparison.png")


# ─────────────────────────────────────────────
# MAIN – Run all tasks
# ─────────────────────────────────────────────

def main():
    # Usage:
    #   python restoration.py                → synthetic surveillance image
    #   python restoration.py street.jpg     → your own image file
    #   python restoration.py webcam         → capture from webcam
    source = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    print(f"\n[INFO] Image source: '{source}'\n")

    # Task 1 – Preprocessing
    print("── Task 1: Image Acquisition & Preprocessing ──────────────")
    gray = preprocess(source)

    # Task 2 – Noise Modeling
    print("\n── Task 2: Noise Modeling ──────────────────────────────────")
    noisy_dict = model_noise(gray)

    # Task 3 – Restoration Filters
    print("\n── Task 3: Restoration Filters (Gaussian noise) ────────────")
    restored_gaussian = apply_filters(noisy_dict["gaussian"], "Gaussian Noise")

    print("\n── Task 3: Restoration Filters (Salt-and-Pepper noise) ─────")
    restored_sp = apply_filters(noisy_dict["salt_pepper"], "Salt-and-Pepper Noise")

    # Tasks 4 & 5 – Evaluation & Discussion
    analytical_discussion(gray, noisy_dict, restored_gaussian, restored_sp)

    print("\n[DONE] All tasks completed. Check the outputs/ folder.")


if __name__ == "__main__":
    main()