"""
Takes images after preprocessing turns it into masks and calculates the overlap between synapsin and FM puncta to put in a csv file. 
=============

Detect synapsin and FM puncta in every renamed field and report the
syn/FM overlap percentage per image.

Pipeline (applied identically to BOTH channels — that is the key):
  1. Rolling-ball background subtraction (radius = ROLLING_BALL_RADIUS px)
  2. Light Gaussian smoothing (sigma = GAUSSIAN_SIGMA)
  3. Per-pixel local z-score against a Z_WINDOW x Z_WINDOW window
  4. Threshold at z > Z_THRESHOLD
  5. Connected components, keeping objects with MIN_AREA <= area <= MAX_AREA px,
     with a circularity filter (>= MIN_CIRCULARITY) and a small variance-based
     mask expansion (MASK_VARIANCE_SIZE). See the SETTINGS block for values.

Output (written into <FOLDER>/puncta_analysis/):
  - overlap_summary.csv      : per-field counts and overlap %
  - field_<N>_overlay.png    : diagnostic overlay (only if SAVE_OVERLAYS=True)
  - masks/IMGsyn_<...>.tif   : binary mask of detected syn puncta (if SAVE_MASKS)
  - masks/IMGFM_<...>.tif    : binary mask of detected FM puncta  (if SAVE_MASKS)
"""

import csv
import argparse
import re
import sys
from pathlib import Path

# ============================================================
#  SETTINGS
# ============================================================

# Where the renamed images live (must contain IMGsyn_*.tif and IMGFM_*.tif).
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "FM_syn"  # repo-relative
INPUT_FOLDER = Path(  # <-- --input-folder command-line default (interactive runs use a picker)
    _DATA_ROOT / "Control_FM_syn" / "renamed_v3"
)

# Where results get written.
OUTPUT_FOLDER = INPUT_FOLDER.parent / "puncta_analysis"

# Condition label used in the filenames (C / H / HR).
CONDITION = "C"

# --- BATCH MODE -------------------------------------------------------------
# When run interactively (no command-line args) an instructions window opens
# first, then a folder picker:
#   BATCH_MODE = True  -> pick a root folder, then tick which renamed_v3 folders
#                         under it to analyse (checks <cond>/renamed_v3 and the
#                         nested <cond>/cs*/renamed_v3 layout).
#   BATCH_MODE = False -> pick and analyse a single renamed_v3 folder.
# INPUT_FOLDER is the --input-folder command-line default. Set PARENT_FOLDER to
# a path to skip the batch folder-picker (leave None to be prompted).
BATCH_MODE = True
PARENT_FOLDER = None   # root data folder for batch mode; None -> ask with a picker

# --- Detection parameters -------------------------------------------------
ROLLING_BALL_RADIUS = 18        # pixels — same for both channels
GAUSSIAN_SIGMA      = 1.2      # post-rolling-ball smoothing
Z_WINDOW            = 70        # local-z window in pixels
Z_THRESHOLD         = 3        # local-z detection threshold (lower = more sensitive)
MIN_AREA            = 15       # px^2 lower size cut (~4 px diameter)
MAX_AREA            = 650      # px^2 upper size cut (excludes cell bodies)

# Slightly enlarge (dilate) each detected punctum mask so puncta aren't clipped.
# This does not change the detection step itself.
MASK_VARIANCE_SIZE  = 4.5

# Shape filter: 4π·area / perimeter²  (1 = perfect circle, ~0.4 = mildly elongated)
# Set to 0.0 to disable. Synaptic puncta are roughly round — 0.3 is liberal
# and just removes very stringy/dendritic shapes.
MIN_CIRCULARITY     = 0.35

# --- Optional outputs -----------------------------------------------------
SAVE_MASKS    = False           # write binary mask TIFFs (one per channel per field)
SAVE_OVERLAYS = True            # write a diagnostic PNG per field
SAVE_STACKS   = True            # write a 4-slice TIFF stack per field
                                # (FM image, FM mask, syn image, syn mask)
# ============================================================

try:
    import numpy as np
    import tifffile
    from scipy.ndimage import gaussian_filter, uniform_filter, label
    from skimage import morphology
    from skimage.morphology import disk, remove_small_objects, remove_small_holes
    from skimage.measure import regionprops, find_contours
except ImportError as e:
    sys.exit(f"Missing package: {e}\nRun:  pip install numpy tifffile scikit-image scipy matplotlib")

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def rolling_ball(img: np.ndarray, radius: int) -> np.ndarray:
    """Fiji-style rolling ball via morphological reconstruction."""
    img = img.astype(np.float32)
    selem = disk(radius)
    background = morphology.reconstruction(morphology.erosion(img, selem), img)
    return np.clip(img - background, 0, None)


def detect_puncta(img: np.ndarray) -> np.ndarray:
    """Return a binary mask of detected puncta using the tuned pipeline."""
    bgsub  = rolling_ball(img, ROLLING_BALL_RADIUS)
    smooth = gaussian_filter(bgsub, GAUSSIAN_SIGMA)

    # Local z-score: per-pixel test against a Z_WINDOW x Z_WINDOW background window
    mu  = uniform_filter(smooth,           size=Z_WINDOW)
    mu2 = uniform_filter(smooth * smooth,  size=Z_WINDOW)
    sd  = np.sqrt(np.maximum(mu2 - mu * mu, 1.0))
    z   = (smooth - mu) / sd

    mask = z > Z_THRESHOLD
    mask = remove_small_objects(mask, min_size=MIN_AREA)
    mask = remove_small_holes(mask, area_threshold=20)

    # Drop anything larger than MAX_AREA (cell bodies, blob artifacts)
    lab, _ = label(mask)
    sizes = np.bincount(lab.ravel())
    too_big = np.where(sizes > MAX_AREA)[0]
    for i in too_big:
        if i == 0:
            continue
        mask[lab == i] = False

    # circularity filter — remove very elongated / dendritic blobs
    if MIN_CIRCULARITY > 0:
        lab, _ = label(mask)
        for prop in regionprops(lab):
            if prop.perimeter == 0:
                continue
            circ = 4 * np.pi * prop.area / (prop.perimeter ** 2)
            if circ < MIN_CIRCULARITY:
                mask[lab == prop.label] = False

    return mask


def enlarge_mask_with_variance(mask: np.ndarray, size: int = MASK_VARIANCE_SIZE) -> np.ndarray:
    """
    Enlarge a binary puncta mask using local variance, similar to Fiji's
    Variance filter behavior on a black/white mask.

    For a binary mask, local variance is >0 only where a window contains both
    foreground and background pixels. Those pixels form a band around puncta.
    Combining that band with the original mask makes the puncta bigger.
    """
    mask = mask.astype(bool)
    if size <= 1:
        return mask

    mask_float = mask.astype(np.float32)
    local_mean = uniform_filter(mask_float, size=size)
    local_mean_sq = uniform_filter(mask_float * mask_float, size=size)
    local_variance = local_mean_sq - local_mean * local_mean

    return mask | (local_variance > 0)


def compute_overlap(syn_mask: np.ndarray, fm_mask: np.ndarray):
    """
    Return (n_syn, n_fm, n_syn_matched, pct_syn_overlap, pct_fm_overlap,
    pct_pixel_overlap).
    A syn (or FM) object counts as matched if ANY of its pixels falls inside
    the other channel's mask.
    """
    syn_lab, n_syn = label(syn_mask)
    fm_lab,  n_fm  = label(fm_mask)
    if n_syn == 0 or n_fm == 0:
        pct_pix = 0.0
    else:
        pct_pix = 100 * (syn_mask & fm_mask).sum() / max(syn_mask.sum(), 1)

    n_syn_matched = sum(1 for i in range(1, n_syn + 1) if fm_mask[syn_lab == i].any())
    n_fm_matched  = sum(1 for i in range(1, n_fm  + 1) if syn_mask[fm_lab  == i].any())
    pct_syn = 100 * n_syn_matched / max(n_syn, 1)
    pct_fm  = 100 * n_fm_matched  / max(n_fm,  1)
    return n_syn, n_fm, n_syn_matched, pct_syn, pct_fm, pct_pix



# ---------------------------------------------------------------------------
# Overlay diagnostic image
# ---------------------------------------------------------------------------

def _draw_contours(ax, mask, color, lw=0.6):
    """Draw the outline of every object in `mask` on `ax` (Fiji-ROI style)."""
    for contour in find_contours(mask.astype(np.float32), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=lw)


def save_overlay(syn_img, fm_img, syn_mask, fm_mask, stats, out_path):
    if not HAS_PLT:
        return

    # 2 rows x 4 cols:
    #   row 1 — raw + mask + raw-with-ROIs for syn
    #   row 2 — raw + mask + raw-with-ROIs for FM
    #   col 4 — combined views (RGB mask overlay, syn raw with both ROIs)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    syn_vmax = np.percentile(syn_img, 99.5)
    fm_vmax  = np.percentile(fm_img,  99.5)

    # --- syn row ---
    axes[0, 0].imshow(syn_img, cmap='gray', vmin=0, vmax=syn_vmax)
    axes[0, 0].set_title("syn (raw)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(syn_mask, cmap='Reds')
    axes[0, 1].set_title(f"syn mask ({stats['n_syn']} objs)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(syn_img, cmap='gray', vmin=0, vmax=syn_vmax)
    _draw_contours(axes[0, 2], syn_mask, color='cyan')
    axes[0, 2].set_title("syn raw + detected ROIs")
    axes[0, 2].axis('off')

    # --- FM row ---
    axes[1, 0].imshow(fm_img, cmap='gray', vmin=0, vmax=fm_vmax)
    axes[1, 0].set_title("FM (raw)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fm_mask, cmap='Greens')
    axes[1, 1].set_title(f"FM mask ({stats['n_fm']} objs)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(fm_img, cmap='gray', vmin=0, vmax=fm_vmax)
    _draw_contours(axes[1, 2], fm_mask, color='yellow')
    axes[1, 2].set_title("FM raw + detected ROIs")
    axes[1, 2].axis('off')

    # --- combined column ---
    rgb = np.zeros((*syn_img.shape, 3))
    rgb[..., 0] = syn_mask
    rgb[..., 1] = fm_mask
    axes[0, 3].imshow(rgb)
    axes[0, 3].set_title(
        f"R=syn  G=FM  Y=overlap\n"
        f"syn∩FM: {stats['pct_syn']:.1f}%   pixels: {stats['pct_pix']:.1f}%"
    )
    axes[0, 3].axis('off')

    # syn raw with BOTH sets of contours, so you can see colocalisation in situ
    axes[1, 3].imshow(syn_img, cmap='gray', vmin=0, vmax=syn_vmax)
    _draw_contours(axes[1, 3], syn_mask, color='cyan')
    _draw_contours(axes[1, 3], fm_mask,  color='yellow')
    axes[1, 3].set_title("syn raw + syn (cyan) + FM (yellow) ROIs")
    axes[1, 3].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, condition=CONDITION):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        sys.exit(f"Input folder not found:\n  {input_folder}")

    output_folder.mkdir(exist_ok=True)
    if SAVE_MASKS:
        (output_folder / "masks").mkdir(exist_ok=True)

    # all synapsin images for this condition, ordered by field number
    syn_files = sorted(
        input_folder.glob(f"IMGsyn_{condition}_*.tif"),
        key=lambda p: int(re.search(r"_(\d+)\.tif$", p.name).group(1)),
    )
    if not syn_files:
        sys.exit(f"No IMGsyn_{condition}_*.tif files found in {input_folder}")

    print(f"Input        : {input_folder}")
    print(f"Output       : {output_folder}")
    print(f"Fields       : {len(syn_files)} (condition {condition})")
    print(f"Params       : Z>{Z_THRESHOLD}  rb_radius={ROLLING_BALL_RADIUS}  "
          f"sigma={GAUSSIAN_SIGMA}  area=[{MIN_AREA},{MAX_AREA}]")
    print()

    # Analyse each field: match its FM image, detect puncta in both channels,
    # then measure how much of the synapsin signal overlaps the FM signal.
    rows = []
    for syn_path in syn_files:
        fld = int(re.search(r"_(\d+)\.tif$", syn_path.name).group(1))   # field number from the filename
        fm_path = input_folder / f"IMGFM_{condition}_{fld}.tif"         # matching FM image for this field
        if not fm_path.exists():
            print(f"Field {fld}: FM file missing, skipping.")
            continue

        syn_img = tifffile.imread(syn_path)                            # load both channels
        fm_img  = tifffile.imread(fm_path)

        # detect puncta in each channel, then slightly enlarge the masks
        syn_mask = enlarge_mask_with_variance(detect_puncta(syn_img))
        fm_mask  = enlarge_mask_with_variance(detect_puncta(fm_img))

        # counts + overlap; pct_syn_with_fm = % of synapsin puncta that are FM+ (presynaptically active)
        n_syn, n_fm, n_match, pct_syn, pct_fm, pct_pix = compute_overlap(syn_mask, fm_mask)
        rows.append({                                                  # one summary row per field
            "field": fld,
            "n_syn": n_syn,
            "n_fm":  n_fm,
            "n_syn_matched": n_match,
            "pct_syn_with_fm": round(pct_syn, 2),
            "pct_fm_with_syn": round(pct_fm,  2),
            "pct_pixel_overlap": round(pct_pix, 2),
        })

        print(f"Field {fld:>3}: syn={n_syn:>4}  FM={n_fm:>4}  "
              f"syn∩FM={pct_syn:>5.1f}%  pixels={pct_pix:>5.1f}%")

        if SAVE_MASKS:
            tifffile.imwrite(
                output_folder / "masks" / f"IMGsyn_{condition}_{fld}_mask.tif",
                syn_mask.astype(np.uint8) * 255,
            )
            tifffile.imwrite(
                output_folder / "masks" / f"IMGFM_{condition}_{fld}_mask.tif",
                fm_mask.astype(np.uint8) * 255,
            )

        if SAVE_OVERLAYS:
            save_overlay(
                syn_img, fm_img, syn_mask, fm_mask,
                stats={"n_syn": n_syn, "n_fm": n_fm,
                       "pct_syn": pct_syn, "pct_pix": pct_pix},
                out_path=output_folder / f"field_{fld:03d}_overlay.png",
            )

        if SAVE_STACKS:
            # 4-slice stack: FM image, FM mask, syn image, syn mask
            # Masks are scaled to 65535 so Fiji shows them at full brightness
            # in the same display range as the raw images.
            stack = np.stack([
                fm_img.astype(np.uint16),
                (fm_mask.astype(np.uint16))  * 65535,
                syn_img.astype(np.uint16),
                (syn_mask.astype(np.uint16)) * 65535,
            ], axis=0)
            tifffile.imwrite(
                output_folder / f"field_{fld:03d}_stack.tif",
                stack,
                imagej=True,
            )

    if not rows:
        sys.exit("No fields were processed successfully.")

    # Write CSV
    csv_path = output_folder / "overlap_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary
    pcts_syn = [r["pct_syn_with_fm"] for r in rows]
    pcts_fm  = [r["pct_fm_with_syn"] for r in rows]
    n_syn    = [r["n_syn"]           for r in rows]
    print()
    print("---- Summary across all fields ----")
    print(f"  syn objects per field:  mean={np.mean(n_syn):.0f}  median={np.median(n_syn):.0f}  "
          f"min={np.min(n_syn)}  max={np.max(n_syn)}")
    print(f"  syn∩FM overlap %:        mean={np.mean(pcts_syn):.1f}%  median={np.median(pcts_syn):.1f}%")
    print(f"  FM∩syn overlap %:        mean={np.mean(pcts_fm):.1f}%  median={np.median(pcts_fm):.1f}%")
    print()
    print(f"CSV saved to: {csv_path}")
    if SAVE_OVERLAYS:
        print(f"Overlays in: {output_folder}")


# ---------------------------------------------------------------------------
# Batch folder discovery + picker GUI
# ---------------------------------------------------------------------------

def detect_condition(folder_name: str) -> str | None:
    name = folder_name.lower()
    if "hypoxia_recovery" in name or "hypoxiarecovery" in name:
        return "HR"
    if "hypoxia" in name:
        return "H"
    if "control" in name or "ctrl" in name:
        return "C"
    return None


def find_renamed_v3_targets(parent: Path):
    """
    Find every renamed_v3 folder under `parent`, preferring per-coverslip
    nesting (<cond>/cs1/renamed_v3, <cond>/cs2/renamed_v3, ...) when present,
    since a condition folder can carry both a stale top-level renamed_v3 from
    before it was split into per-coverslip subfolders AND the current cs*
    ones -- in that case the nested ones are what's current.
    Falls back to a direct <cond>/renamed_v3 only if no cs* subfolders have
    their own renamed_v3.
    Returns a list of dicts: label, input_folder, output_folder, condition.
    """
    if not parent.exists():
        sys.exit(f"\nParent folder not found:\n  {parent}")

    targets = []
    for cond_folder in sorted(parent.iterdir()):
        if not cond_folder.is_dir():
            continue
        condition = detect_condition(cond_folder.name)
        if condition is None:
            continue

        nested_found = False
        for sub in sorted(cond_folder.iterdir()):
            if not sub.is_dir():
                continue
            nested = sub / "renamed_v3"
            if nested.is_dir():
                targets.append({
                    "label": f"{cond_folder.name}/{sub.name}",
                    "input_folder": nested,
                    "output_folder": sub / "puncta_analysis",
                    "condition": condition,
                })
                nested_found = True

        if not nested_found:
            direct = cond_folder / "renamed_v3"
            if direct.is_dir():
                targets.append({
                    "label": cond_folder.name,
                    "input_folder": direct,
                    "output_folder": cond_folder / "puncta_analysis",
                    "condition": condition,
                })

    return targets


def select_targets_gui(targets: list[dict]) -> list[dict]:
    """Checkbox window letting the user pick which targets to analyze."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Select folders to analyze")

    tk.Label(root, text="Select the renamed_v3 folders to analyze:",
             padx=10, pady=10).pack(anchor="w")

    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(fill="both", expand=True, padx=10)
    canvas = tk.Canvas(canvas_frame, height=min(400, 24 * len(targets) + 10))
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    checklist_frame = ttk.Frame(canvas)

    checklist_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=checklist_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    vars_ = []
    for t in targets:
        var = tk.BooleanVar(value=True)
        ttk.Checkbutton(checklist_frame, text=f"{t['label']}  ({t['condition']})",
                        variable=var).pack(anchor="w")
        vars_.append(var)

    selected: list[dict] = []

    def on_run():
        selected.extend(t for t, v in zip(targets, vars_) if v.get())
        root.destroy()

    def on_cancel():
        root.destroy()

    button_frame = ttk.Frame(root)
    button_frame.pack(fill="x", padx=10, pady=10)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(button_frame, text="Run", command=on_run).pack(side="right", padx=5)

    root.mainloop()

    if not selected:
        sys.exit("\nNo folders selected. Exiting.")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Run puncta overlap analysis on one renamed_v3 folder.")
    parser.add_argument("--input-folder", type=Path, default=INPUT_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--condition", default=CONDITION, choices=("C", "H", "HR"))
    return parser.parse_args()


def pick_directory(title):
    """Open a folder-selection dialog and return the chosen directory."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askdirectory(title=title)
    root.destroy()
    if not chosen:
        sys.exit("No folder selected.")
    return Path(chosen)


def show_message(message, title="FM puncta analysis"):
    """Pop up an information window with instructions before the folder picker."""
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


def condition_from_path(folder):
    """Infer the condition (C/H/HR) from the chosen folder's path; fall back to CONDITION."""
    for name in [folder.name] + [p.name for p in folder.parents]:
        c = detect_condition(name)
        if c:
            return c
    return CONDITION


if __name__ == "__main__":
    # Three ways to run: command-line args, batch (many folders), or one folder.
    if len(sys.argv) > 1:
        # --- Command-line mode: no dialogs, use the given arguments ---
        args = parse_args()
        # write results next to the input folder unless --output-folder was given
        out = args.output_folder if args.output_folder is not None else args.input_folder.parent / "puncta_analysis"
        main(args.input_folder, out, args.condition)
        print()
    elif BATCH_MODE:
        # --- Batch: pick a root folder, find every renamed_v3 under it, choose which to run ---
        if PARENT_FOLDER is None:
            show_message(                                    # instructions pop-up
                "Batch mode is ON, so you can analyse several folders at once.\n\n"
                "1. In the next window, choose the ROOT data folder that contains the\n"
                "   condition folders (each with a renamed_v3 folder inside).\n"
                "2. Then tick the folders you want to analyse and click OK.",
                "Batch mode")
            parent = pick_directory("Select the ROOT data folder (contains the condition folders)")
        else:
            parent = PARENT_FOLDER                           # skip the dialog if a path was set
        # each target = one renamed_v3 folder + its detected condition + output path
        targets = find_renamed_v3_targets(parent)
        if not targets:
            sys.exit(f"\nNo renamed_v3 folders found under:\n  {parent}")
        targets = select_targets_gui(targets)                # tick-box window: keep only chosen ones
        print(f"\nBATCH MODE — {len(targets)} folder(s) to analyze:")
        for t in targets:
            print(f"   • {t['label']}")
        # analyse each chosen folder in turn
        for t in targets:
            print(f"\n========== {t['label']} ==========")
            main(t["input_folder"], t["output_folder"], t["condition"])
        print(f"\nAll {len(targets)} folder(s) done.")
    else:
        # --- Single folder: pick one and analyse it ---
        show_message(
            "Batch mode is OFF, so this run analyses ONE folder.\n\n"
            "Choose the single renamed_v3 folder to analyse in the next window.",
            "Single-folder mode")
        folder = pick_directory("Select the renamed_v3 folder to analyse")
        # condition (C/H/HR) is auto-detected from the folder name
        main(folder, folder.parent / "puncta_analysis", condition_from_path(folder))
        print()
