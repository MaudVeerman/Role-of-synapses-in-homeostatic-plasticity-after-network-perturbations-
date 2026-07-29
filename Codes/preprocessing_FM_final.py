"""
Microscopy Image Renamer & Converter
======================================

Pipeline (per field of 2-3 images):
  1. GROUP raw files into fields using SYN as the delimiter (see group_by_syn):
       G/R ratio cleanly separates syn (G/R ~0.0) from FM/trans (G/R ~0.7)
       regardless of exposure. A new syn image always starts a new field.
  2. CLASSIFY channels within each group (see classify_field):
       syn -> image with G/R ratio < 0.1  (essentially pure red)
       FM  -> remaining image with HIGHEST red std (most contrast). A
              RELATIVE comparison against the other image(s) still in the
              group, not a fixed absolute threshold, so it works regardless
              of how bright or dim that particular coverslip's acquisition was.
       t   -> the leftover, if a third (transmitted-light) image is present.
              Occasionally missing. If only syn + one other image were
              captured for a field, that other image is assumed to be FM
              (trans is the channel most often skipped, not FM).
  3. LOAD each image and SUM RGB -> single-channel grayscale (float32)
  3. STRETCH all channels to full 16-bit range (0..65535)
       Percentile-based: lo = p0.1, hi = p99.8 — robust to hot pixels.
       Done first so the aligner has plenty of signal to work with.
  4. ALIGN FM + syn via Fiji's StackReg plugin, "Rigid Body" mode
       (translation + rotation), FM as reference. t is NOT aligned.
       NO rolling ball before this step — syn keeps its full structure
       so StackReg can find rotation as well as translation.
  5. ROLLING BALL background subtraction on syn only (post-align).
  6. SAVE as 16-bit TIFF: IMG<channel>_<condition>_<field>.tif

Caching:
  After the first alignment pass, each field's aligned syn is written to
  <OUTPUT_SUBDIR>/aligned_cache/aligned_syn_<COND>_<field>.tif.
  Set SKIP_ALIGNMENT=True on later runs to read those TIFFs back and skip
  Fiji entirely (useful when iterating on rolling-ball / stretch settings).

Output naming:
    IMGsyn_C_1.tif, IMGFM_C_1.tif, IMGt_C_1.tif, IMGsyn_C_2.tif, ...
    where condition is C (control), H (hypoxia), or HR (hypoxia recovery),
    auto-detected from the folder name.
"""

from __future__ import annotations

import re
import sys
import subprocess
import tempfile
from pathlib import Path

# ============================================================
#  SETTINGS
# ============================================================

# --- WHAT TO PROCESS -----------------------------------------------------
# Two modes:
#   BATCH_MODE = False -> process FOLDER only.
#   BATCH_MODE = True  -> process EVERY immediate subfolder of PARENT_FOLDER
#                         that contains image files. Each subfolder gets its
#                         own renamed_v3/ + aligned_cache/ inside it.
BATCH_MODE = True

# Used when BATCH_MODE = True. Leave as None to pick the folder from a pop-up
# dialog when the script runs, or set a path here to skip the dialog.
PARENT_FOLDER = None   # root data folder (BATCH_MODE = True)

# Used when BATCH_MODE = False. Same idea: None -> choose from a dialog.
FOLDER = None          # single folder (BATCH_MODE = False)

# Output goes to <FOLDER>/<OUTPUT_SUBDIR>
OUTPUT_SUBDIR = "renamed_v3"

CHANNELS_PER_FIELD = 3
DRY_RUN = False

# Channel classification: syn is the image whose green channel is essentially
# zero (G/R < threshold). syn sits at G/R ~0-0.03 and FM/trans at ~0.7-0.8, so
# any cutoff in between works. FM vs trans is then decided by relative red-std
# within each field (higher std = FM), not an absolute cutoff.
SYN_GR_THRESHOLD = 0.1

# Stage-2 phase-correlation refinement (see refine_alignment_xcorr) is only
# trusted up to this many pixels of measured shift. A larger shift means the
# two channels don't agree well enough to refine, so the field's alignment
# is treated as unreliable and the field is skipped.
ALIGN_MAX_SHIFT_PX = 8.0

# Rolling ball radius (pixels) applied to the syn channel
SYN_ROLLING_BALL_RADIUS = 34

# Stretch percentiles — applied to every channel to map values into 0..65535.
# 0.1 and 99.9 use ~99.8% of the histogram while clipping only the most extreme
# hot/cold pixels, so the output uses the full 16-bit range without being
# wrecked by outliers.
STRETCH_LOW_PERCENTILE  = 0.1
STRETCH_HIGH_PERCENTILE = 99.8

# Path to your Fiji executable (used for StackReg Rigid Body alignment).
# macOS default below — adjust if Fiji lives elsewhere.
FIJI_PATH = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx")  # <-- EDIT: path to your Fiji executable

# When True, skip Fiji entirely and reuse the per-field shifts from a previous run.
# Set to True after one slow alignment pass so you can iterate on filtering fast.
SKIP_ALIGNMENT = False

# Where the cached aligned syn TIFFs live (inside <FOLDER>/<OUTPUT_SUBDIR>).
ALIGNMENT_CACHE_DIR_NAME = "aligned_cache"

# ============================================================

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed. Run:  pip install Pillow")

try:
    import tifffile
except ImportError:
    sys.exit("tifffile not installed. Run:  pip install tifffile")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not installed. Run:  pip install numpy")

try:
    from skimage import morphology
    from skimage.morphology import disk
except ImportError:
    sys.exit("scikit-image not installed. Run:  pip install scikit-image")

try:
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as ndshift
    HAS_ALIGNMENT = True
except Exception:
    HAS_ALIGNMENT = False
    print("Warning: scikit-image registration / scipy not available — alignment disabled.")


def refine_alignment_xcorr(reference: np.ndarray, moving: np.ndarray,
                           max_shift_px: float = ALIGN_MAX_SHIFT_PX):
    """
    Stage 2 — sub-pixel translational refinement via phase cross-correlation.
    Fast and accurate for unimodal pairs, but can be wrong when the two
    channels look very different (different intensity distributions /
    different structures visible) — in that case the measured shift comes
    out larger than max_shift_px and is rejected rather than applied.

    Returns (result, dy, dx, applied). When applied is False, result is the
    original `moving` array, unchanged, and the field should be treated as
    unreliably aligned.
    """
    if not HAS_ALIGNMENT:
        return moving, 0.0, 0.0, False
    shift, _, _ = phase_cross_correlation(reference, moving, upsample_factor=50)
    dy, dx = float(shift[0]), float(shift[1])
    if abs(dy) > max_shift_px or abs(dx) > max_shift_px:
        return moving, dy, dx, False
    refined = ndshift(moving, (dy, dx), order=3, mode="constant", cval=0)
    return refined, dy, dx, True

INPUT_EXTENSIONS = {".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OUTPUT_EXT = ".tif"


# ---------------------------------------------------------------------------
# Channel classification
# ---------------------------------------------------------------------------

def image_stats(path: Path) -> tuple[float, float]:
    """Return (G/R ratio, red-channel std-dev) for channel classification."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    r_mean = r.mean()
    g_r = float(g.mean() / r_mean) if r_mean > 0 else 0.0
    return g_r, float(r.std())


def group_by_syn(file_list):
    """
    Group raw acquisition files into per-field chunks using SYN as the
    natural delimiter, instead of guessing "transmitted light" from an
    absolute pixel-brightness cutoff (that broke down on dim coverslips --
    see the note by SYN_GR_THRESHOLD above).

    G/R ratio reliably separates syn (~0.0) from FM/trans (~0.7) regardless
    of exposure. Every field has exactly one syn image, but syn is NOT
    always the first image acquired for that field -- a field can be
    FM, syn, trans just as easily as syn, FM, trans. So a new group is only
    started when the CURRENT group already contains a syn and *another* syn
    arrives (that's the reliable signal a new field has begun); a leading
    non-syn image is left in place to be picked up by whichever syn follows
    it. Groups are also capped at CHANNELS_PER_FIELD: once that many images
    have accumulated, the next one always starts a fresh group (handles the
    case where a field's own syn was skipped entirely, and/or a run of
    non-syn images from two adjacent incomplete fields would otherwise merge).

    Returns (groups, stats) where stats is {path: (g_r, r_std)}, reused by
    classify_field so each file's pixel stats are only computed once.
    """
    stats = {f: image_stats(f) for f in file_list}
    groups = []
    current = []
    current_has_syn = False
    for f in file_list:
        is_syn = stats[f][0] < SYN_GR_THRESHOLD
        if (is_syn and current_has_syn) or len(current) >= CHANNELS_PER_FIELD:
            groups.append(current)
            current = [f]
            current_has_syn = is_syn
        else:
            current.append(f)
            current_has_syn = current_has_syn or is_syn
    if current:
        groups.append(current)
    return groups, stats


def classify_field(group, stats):
    """Decide which file in the group (1-3 images) is syn / FM / t.

    `stats` is the {path: (g_r, r_std)} dict from group_by_syn, so pixel
    stats aren't recomputed here.

    Returns (result, missing). `missing` lists which of 'syn' / 'FM' couldn't
    be reliably identified in this group -- e.g. no image had a low enough
    G/R to count as syn, or there were no non-syn images left for FM. Both
    syn and FM are required downstream, so callers should skip the field
    when `missing` is non-empty.
    """
    by_gr = sorted(group, key=lambda p: stats[p][0])
    syn_path = by_gr[0]
    remaining = [p for p in group if p != syn_path]

    missing = []
    result = {}

    if stats[syn_path][0] >= SYN_GR_THRESHOLD:
        missing.append("syn")
    else:
        result[syn_path] = "syn"

    # FM = the remaining image with the higher red std (relative, not a fixed
    # threshold). If only one non-syn image is present, assume it's FM.
    fm_path = max(remaining, key=lambda p: stats[p][1]) if remaining else None

    if fm_path is None:
        missing.append("FM")
    else:
        result[fm_path] = "FM"
        t_candidates = [p for p in remaining if p != fm_path]
        t_path = t_candidates[0] if t_candidates else None
        if t_path:
            result[t_path] = "t"

    return result, missing


# ---------------------------------------------------------------------------
# Loading: RGB -> grayscale sum, float32
# ---------------------------------------------------------------------------

def load_grayscale_sum(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]
    return arr


# ---------------------------------------------------------------------------
# Rolling ball background subtraction (Fiji-style)
# ---------------------------------------------------------------------------

def rolling_ball_subtract(arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Morphological-reconstruction rolling ball, equivalent to the Fiji plugin.
    Estimates the slow-varying background by eroding with a disk of given
    radius then reconstructing under the original image, and subtracts it.
    """
    arr = arr.astype(np.float32)
    selem = disk(radius)
    eroded = morphology.erosion(arr, selem)
    background = morphology.reconstruction(eroded, arr)
    return np.clip(arr - background, 0, None)


# ---------------------------------------------------------------------------
# Percentile stretch to full 16-bit range
# ---------------------------------------------------------------------------

def stretch_to_uint16(arr: np.ndarray,
                      low: float = STRETCH_LOW_PERCENTILE,
                      high: float = STRETCH_HIGH_PERCENTILE) -> np.ndarray:
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi > lo:
        out = np.clip((arr - lo) / (hi - lo), 0, 1) * 65535.0
    else:
        out = np.zeros_like(arr)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Alignment via Fiji's StackReg plugin (Rigid Body)
# ---------------------------------------------------------------------------

def align_stack_with_fiji(channels: dict) -> dict:
    """
    Stack the channel arrays, run Fiji's StackReg plugin in Rigid Body mode
    (translation + rotation) using FM as the reference slice, then return
    the de-stacked aligned arrays.
    """
    if not FIJI_PATH.exists():
        sys.exit(f"\nFiji not found at: {FIJI_PATH}\nEdit FIJI_PATH at the top of the script.")

    # FM is the alignment reference -> put it first in the stack.
    # StackReg aligns each slice to the previous one, so slice-1 (FM) stays
    # fixed and slice-2 (syn) gets transformed to match it.
    order = ["FM"] + [c for c in channels if c != "FM"]
    stack = np.stack([channels[c].astype(np.float32) for c in order], axis=0)

    # We deliberately do NOT use a context manager so the temp dir survives
    # for debugging if Fiji fails.
    td = Path(tempfile.mkdtemp(prefix="renamer_align_"))
    in_tif  = td / "in.tif"
    out_tif = td / "out.tif"
    macro   = td / "align.ijm"

    # imagej=True writes an ImageJ-native multi-slice TIFF so Fiji's Opener
    # handles it directly and doesn't fall through to Bio-Formats (which has
    # a VerifyError under modern macOS Java).
    tifffile.imwrite(in_tif, stack.astype(np.float32), imagej=True)

    # Macro: open stack, run StackReg Rigid Body, save the aligned (active)
    # image, then exit. Uses "StackReg " (trailing space) since that's the
    # exact plugin command name in most Fiji installs; if your Fiji version
    # errors on this line, try removing the trailing space instead.
    macro.write_text(
        f'open("{in_tif.as_posix()}");\n'
        'run("StackReg ", "transformation=[Rigid Body]");\n'
        f'saveAs("Tiff", "{out_tif.as_posix()}");\n'
        'eval("script", "System.exit(0);");\n'
    )

    # --console makes Fiji print plugin errors to stdout/stderr.
    result = subprocess.run(
        [str(FIJI_PATH), "--ij2", "--headless", "--console",
         "-macro", str(macro)],
        capture_output=True, text=True, timeout=300,
    )

    if not out_tif.exists():
        msg = [
            "",
            "Fiji did NOT produce the aligned stack.",
            f"  Temp dir kept for inspection: {td}",
            f"  Return code: {result.returncode}",
            "  --- Fiji STDOUT ---",
            result.stdout or "(empty)",
            "  --- Fiji STDERR ---",
            result.stderr or "(empty)",
        ]
        sys.exit("\n".join(msg))

    aligned = tifffile.imread(out_tif).astype(np.float32)

    # Cleanup on success
    import shutil
    shutil.rmtree(td, ignore_errors=True)

    return {order[i]: aligned[i] for i in range(len(order))}


# ---------------------------------------------------------------------------
# Condition detection from folder name
# ---------------------------------------------------------------------------

def detect_condition(folder_name: str, prompt_if_unknown: bool = True) -> str | None:
    name = folder_name.lower()
    if "hypoxia_recovery" in name or "hypoxiarecovery" in name:
        return "HR"
    if "hypoxia" in name:
        return "H"
    if "control" in name or "ctrl" in name:
        return "C"
    if not prompt_if_unknown:
        return None
    print(f"\nCould not detect condition from folder name: '{folder_name}'")
    print("C = control  |  H = hypoxia  |  HR = hypoxia_recovery")
    while True:
        choice = input("Enter condition [C / H / HR]: ").strip().upper()
        if choice in ("C", "H", "HR"):
            return choice
        print("Please enter C, H, or HR.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_folder(folder: Path, condition_source: Path | None = None) -> None:
    if not folder.exists():
        sys.exit(f"\nFolder not found:\n  {folder}\nCheck the FOLDER path.")

    condition = detect_condition((condition_source or folder).name)
    condition_label = {"C": "control", "H": "hypoxia", "HR": "hypoxia recovery"}[condition]

    print(f"\nFolder       : {folder}")
    print(f"Condition    : {condition}  ({condition_label})")
    print(f"Group size   : {CHANNELS_PER_FIELD} images per field")
    print(f"Rolling ball : radius {SYN_ROLLING_BALL_RADIUS} (syn only)")
    print(f"Stretch      : percentile {STRETCH_LOW_PERCENTILE} .. {STRETCH_HIGH_PERCENTILE} -> 0..65535")
    print("Alignment    : Fiji StackReg Rigid Body + phase-xcorr refinement — FM reference")
    print(f"Dry-run      : {'YES -- no files will be written' if DRY_RUN else 'NO -- files will be saved'}")

    def seq_num(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    files = sorted(
        (f for f in folder.iterdir()
         if f.is_file() and f.suffix.lower() in INPUT_EXTENSIONS),
        key=seq_num,
    )

    if not files:
        sys.exit(f"\nNo image files found in:\n  {folder}")

    print(f"\nFound {len(files)} image files.")

    # Group by content (see group_by_syn): a new syn image starts a new
    # field, so a missing transmitted-light image just yields a 2-image
    # group instead of corrupting the fields around it.
    field_groups, field_stats = group_by_syn(files)
    n_partial  = sum(1 for g in field_groups if 1 < len(g) < CHANNELS_PER_FIELD)
    n_single   = sum(1 for g in field_groups if len(g) == 1)
    if n_partial:
        print(f"  Note: {n_partial} field(s) with a missing image detected "
              f"(gap in sequence) — will process syn+FM only for those.")
    if n_single:
        print(f"  Warning: {n_single} isolated single image(s) found — will be skipped.")

    out_dir = folder / OUTPUT_SUBDIR
    if not DRY_RUN:
        out_dir.mkdir(exist_ok=True)
        print(f"Output       : {out_dir}")

    # Aligned-syn cache directory (one TIFF per field) — so SKIP_ALIGNMENT
    # runs can reuse StackReg's output without invoking Fiji again.
    cache_dir = out_dir / ALIGNMENT_CACHE_DIR_NAME
    if not DRY_RUN:
        cache_dir.mkdir(exist_ok=True)
    n_cached = len(list(cache_dir.glob("aligned_syn_*.tif"))) if cache_dir.exists() else 0
    if SKIP_ALIGNMENT and n_cached == 0:
        sys.exit("\nSKIP_ALIGNMENT=True but no cached aligned-syn TIFFs found in "
                 f"{cache_dir}. Run once with SKIP_ALIGNMENT=False first.")
    print(f"Aligned cache: {n_cached} TIFFs in {cache_dir}")
    print(f"Skip Fiji?   : {'YES (using cache)' if SKIP_ALIGNMENT else 'NO (will run Fiji)'}")
    print()

    field_num    = 1
    rename_count = 0
    warnings     = []

    for group in field_groups:
        if len(group) == 1:
            print(f"Skipping field {field_num}: only 1 image, cannot classify channels.")
            field_num += 1
            continue

        seq_range = f"{group[0].stem} .. {group[-1].stem}"
        print(f"Field {field_num}  [{seq_range}]")

        channel_map, missing = classify_field(group, field_stats)

        if missing:
            msg = (f"  Field {field_num}: could not reliably identify "
                   f"{'/'.join(missing)} in this group — skipping "
                   f"(syn and FM are both required).")
            print(msg)
            warnings.append(msg)
            field_num += 1
            continue

        # Step 2: load all 3 channels (raw grayscale, no rolling ball yet)
        raw = {}
        for img_path in group:
            ch = channel_map[img_path]
            g_r, r_std = field_stats[img_path]
            print(f"  {img_path.name:<22}  G/R={g_r:.2f}  R_std={r_std:.2f}"
                  f"  ->  IMG{ch}_{condition}_{field_num}{OUTPUT_EXT}")
            raw[ch] = load_grayscale_sum(img_path)

        # Step 3: stretch every channel into the full 16-bit range
        # (no rolling ball yet — StackReg needs the full syn structure)
        arrays = {ch: stretch_to_uint16(a) for ch, a in raw.items()}

        # Step 4: alignment — load cached aligned syn OR run StackReg now,
        # then ALWAYS run a sub-pixel phase-correlation refinement so any
        # residual translation that StackReg missed is cleaned up.
        aligned_syn_path = cache_dir / f"aligned_syn_{condition}_{field_num}.tif"
        if "FM" in arrays and "syn" in arrays:
            if aligned_syn_path.exists():
                arrays["syn"] = tifffile.imread(aligned_syn_path).astype(np.float32)
                print(f"    loaded cached aligned syn  ({aligned_syn_path.name})")
            elif SKIP_ALIGNMENT:
                sys.exit(
                    f"\nSKIP_ALIGNMENT=True but no cached aligned syn for field "
                    f"{field_num} ({aligned_syn_path.name}). Set SKIP_ALIGNMENT=False "
                    "to align it, or run again once it's been cached."
                )
            else:
                if not DRY_RUN:
                    print("    aligning FM + syn via Fiji StackReg (Rigid Body)...")
                    aligned = align_stack_with_fiji(
                        {"FM": arrays["FM"], "syn": arrays["syn"]}
                    )
                    arrays["syn"] = aligned["syn"]

            # Stage-2 refinement: phase cross-correlation residual cleanup
            arrays["syn"], dy2, dx2, refined_applied = refine_alignment_xcorr(
                arrays["FM"], arrays["syn"]
            )
            residual = (dy2 * dy2 + dx2 * dx2) ** 0.5

            if not refined_applied:
                msg = (f"  Field {field_num}: alignment still off by "
                       f"{residual:.2f} px after stage 2 "
                       f"(threshold {ALIGN_MAX_SHIFT_PX} px) — skipping this field.")
                print(f"    stage 2 residual: dy={dy2:+.3f}  dx={dx2:+.3f}  "
                      f"|d|={residual:.3f} px  <-- refinement NOT applied, shift too large")
                print(msg)
                warnings.append(msg)
                field_num += 1
                continue

            print(f"    stage 2 residual: dy={dy2:+.3f}  dx={dx2:+.3f}  "
                  f"|d|={residual:.3f} px  <-- refinement applied")

            # Persist the fully-aligned (StackReg + stage 2) syn
            if not DRY_RUN and not aligned_syn_path.exists():
                tifffile.imwrite(
                    aligned_syn_path,
                    np.clip(arrays["syn"], 0, 65535).astype(np.uint16),
                )
                print(f"    cached aligned syn -> {aligned_syn_path.name}")

        # Step 5: rolling ball on aligned syn (post-align)
        if "syn" in arrays and SYN_ROLLING_BALL_RADIUS > 0:
            arrays["syn"] = rolling_ball_subtract(arrays["syn"], SYN_ROLLING_BALL_RADIUS)

        # Step 6: save
        if not DRY_RUN:
            for img_path in group:
                ch = channel_map[img_path]
                new_name = f"IMG{ch}_{condition}_{field_num}{OUTPUT_EXT}"
                arr = np.clip(arrays[ch], 0, 65535).astype(np.uint16)
                tifffile.imwrite(out_dir / new_name, arr)
                rename_count += 1

        field_num += 1

    print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(w)
        print()

    if DRY_RUN:
        print("Dry-run complete. Set DRY_RUN = False and run again to save files.")
    else:
        print(f"Done! {rename_count} files saved to:\n  {out_dir}")


def select_folders_gui(folders: list[Path]) -> list[Path]:
    """
    Show a checkbox window listing `folders` (by name) and let the user pick
    which ones to process. Returns the selected Paths, or exits if the user
    closes the window / clicks Cancel without selecting anything.
    """
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Select folders to process")

    tk.Label(root, text="Select the folders to process:", padx=10, pady=10).pack(anchor="w")

    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(fill="both", expand=True, padx=10)
    canvas = tk.Canvas(canvas_frame, height=min(400, 24 * len(folders) + 10))
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
    for folder in folders:
        var = tk.BooleanVar(value=True)
        ttk.Checkbutton(checklist_frame, text=folder.name, variable=var).pack(anchor="w")
        vars_.append(var)

    selected: list[Path] = []

    def on_run():
        selected.extend(f for f, v in zip(folders, vars_) if v.get())
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


SKIP_SUBFOLDER_NAMES = {OUTPUT_SUBDIR, ALIGNMENT_CACHE_DIR_NAME,
                        "renamed", "renamed_1", "renamed_v2",
                        "masks", "rois", "puncta_analysis",
                        "puncta_analysis_v2", "puncta_analysis_intensity",
                        "puncta_analysis_synfirst"}


def _has_direct_images(folder: Path) -> bool:
    return any(
        f.is_file() and f.suffix.lower() in INPUT_EXTENSIONS
        for f in folder.iterdir()
    )


def nested_image_subfolders(folder: Path) -> list[Path]:
    """Immediate subfolders of `folder` (e.g. cs1, cs2, ...) that contain images directly."""
    out = []
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir() or sub.name in SKIP_SUBFOLDER_NAMES:
            continue
        if _has_direct_images(sub):
            out.append(sub)
    return out


def find_subfolders_with_images(parent: Path):
    """
    Return immediate subfolders of `parent` that contain at least one image,
    either directly inside them, or one level deeper (e.g. <folder>/cs1/*.tif).
    """
    if not parent.exists():
        sys.exit(f"\nParent folder not found:\n  {parent}")
    out = []
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir() or sub.name in SKIP_SUBFOLDER_NAMES:
            continue
        if _has_direct_images(sub) or nested_image_subfolders(sub):
            out.append(sub)
    return out


def pick_directory(title: str) -> Path:
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


def show_message(message, title="FM preprocessing"):
    """Pop up an information window with instructions before the folder picker."""
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


if __name__ == "__main__":
    # Work out which folder(s) to process, then run process_folder() on each.
    if BATCH_MODE:
        # --- Batch: choose a root folder, then pick which sub-folders to process ---
        if PARENT_FOLDER is None:
            show_message(                                    # instructions pop-up
                "Batch mode is ON, so you can process several folders at once.\n\n"
                "1. In the next window, choose the ROOT data folder that contains\n"
                "   the folders you want to process.\n"
                "2. Then tick the folders you want and click OK.",
                "Batch mode")
            parent = pick_directory("Select the ROOT data folder (contains the folders to process)")
        else:
            parent = PARENT_FOLDER                           # skip the dialog if a path was set
        subs = find_subfolders_with_images(parent)           # every sub-folder that holds images
        if not subs:
            sys.exit(f"\nNo image-containing subfolders found in:\n  {parent}")
        subs = select_folders_gui(subs)                      # tick-box window: keep only chosen ones
        print(f"\nBATCH MODE — {len(subs)} folder(s) to process:")
        for s in subs:
            print(f"   • {s.name}")
        # Process each chosen folder in turn
        for s in subs:
            if _has_direct_images(s):
                # images sit directly in this folder -> process it as one set
                print(f"\n========== {s.name} ==========")
                process_folder(s)
            else:
                # no direct images: this folder is split into cs1/cs2/... sub-folders,
                # so process each cs* on its own, taking the condition from `s`'s name
                cs_folders = nested_image_subfolders(s)
                print(f"\n========== {s.name} ({len(cs_folders)} subfolder(s)) ==========")
                for cs in cs_folders:
                    print(f"\n----- {s.name}/{cs.name} -----")
                    process_folder(cs, condition_source=s)
        print(f"\nAll {len(subs)} folder(s) done.")
    else:
        # --- Single folder: choose one folder and process it ---
        if FOLDER is None:
            show_message(
                "Batch mode is OFF, so this run processes ONE folder.\n\n"
                "Choose the single folder to process in the next window.",
                "Single-folder mode")
            folder = pick_directory("Select the folder to process")
        else:
            folder = FOLDER                                  # skip the dialog if a path was set
        process_folder(folder)
    input("\nPress Enter to close...")                       # keep the window open until Enter
