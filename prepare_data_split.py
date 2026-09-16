from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SEED = 2026
N_FOLDS = 5
SEARCH_CANDIDATES = 5000
LABEL_NAMES = {
    0: "Ignore",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Vegetation",
    7: "Agricultural",
    8: "Vehicle",
}


def read_mask_counts(mask_paths: list[Path]) -> np.ndarray:
    counts = np.zeros((len(mask_paths), 9), dtype=np.int64)
    for index, path in enumerate(mask_paths):
        with Image.open(path) as mask:
            values = np.asarray(mask)
        bincount = np.bincount(values.reshape(-1), minlength=9)
        counts[index] = bincount[:9]
    return counts


def build_stratification_features(counts: np.ndarray) -> np.ndarray:
    """Balance class presence and medium/large-area examples."""
    features: list[np.ndarray] = []
    for label in range(9):
        values = counts[:, label]
        positive = values[values > 0]
        features.append(values > 0)
        if len(positive):
            median, upper_quartile = np.quantile(positive, [0.5, 0.75])
            features.append(values >= median)
            features.append(values >= upper_quartile)
    return np.stack(features, axis=1).astype(np.int64)


def choose_folds(counts: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Search equal-sized splits and retain the best class/area balance."""
    sample_count = len(counts)
    fold_sizes = np.full(N_FOLDS, sample_count // N_FOLDS, dtype=np.int64)
    fold_sizes[: sample_count % N_FOLDS] += 1
    base_folds = np.repeat(np.arange(N_FOLDS), fold_sizes)
    expected = fold_sizes / sample_count

    total_pixels = counts.sum(axis=0).astype(np.float64)
    total_features = features.sum(axis=0).astype(np.float64)
    rng = np.random.default_rng(SEED)
    best_score = float("inf")
    best_assignment: np.ndarray | None = None

    for _ in range(SEARCH_CANDIDATES):
        assignment = base_folds[rng.permutation(sample_count)]
        pixel_fractions = np.stack(
            [counts[assignment == fold].sum(axis=0) / total_pixels for fold in range(N_FOLDS)]
        )
        feature_fractions = np.stack(
            [
                features[assignment == fold].sum(axis=0)
                / np.maximum(total_features, 1)
                for fold in range(N_FOLDS)
            ]
        )
        pixel_error = np.abs(pixel_fractions - expected[:, None])
        feature_error = np.abs(feature_fractions - expected[:, None])
        # Pixel balance matters most for mIoU; feature balance stabilizes rare classes.
        score = (
            3.0 * float(pixel_error.max())
            + 1.5 * float(np.quantile(pixel_error, 0.9))
            + float(feature_error.max())
            + 0.5 * float(np.quantile(feature_error, 0.9))
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    assert best_assignment is not None
    return best_assignment


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_split_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate filenames in {path}: {duplicates[:5]}")
    return names


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build the project data/ directory from the official raw dataset "
            "using fixed split txt files."
        )
    )
    parser.add_argument(
        "--raw-root",
        default=str(workspace / "2026-低空图像语义分割赛道-训练集"),
        help="Official extracted dataset root that contains train/train and test_1.",
    )
    parser.add_argument(
        "--out-root",
        default=str(workspace / "data"),
        help="Output data directory to create.",
    )
    parser.add_argument(
        "--split-dir",
        default=str(workspace / "data_splits"),
        help="Directory containing fold_<N>_train.txt and fold_<N>_val.txt.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Validation fold id. Default: 0.",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help=(
            "Use hard links when possible to save disk space. If hardlink fails, "
            "the script automatically falls back to copy."
        ),
    )
    return parser.parse_args()


def summarize(counts: np.ndarray, assignment: np.ndarray) -> dict[str, object]:
    total = counts.sum(axis=0)
    summary: dict[str, object] = {}
    for fold in range(N_FOLDS):
        selected = assignment == fold
        fold_counts = counts[selected]
        fold_pixels = fold_counts.sum(axis=0)
        summary[f"fold_{fold}"] = {
            "images": int(selected.sum()),
            "classes": {
                str(label): {
                    "name": LABEL_NAMES[label],
                    "images_present": int((fold_counts[:, label] > 0).sum()),
                    "pixels": int(fold_pixels[label]),
                    "share_of_all_class_pixels_pct": round(
                        float(fold_pixels[label] / total[label] * 100), 4
                    ),
                }
                for label in range(9)
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    source = Path(args.raw_root).expanduser().resolve()
    train_images = source / "train" / "train" / "images"
    train_masks = source / "train" / "train" / "masks"
    test_images = source / "test_1" / "images"
    output = Path(args.out_root).expanduser().resolve()
    split_source = Path(args.split_dir).expanduser().resolve()
    train_split = split_source / f"fold_{args.fold}_train.txt"
    val_split = split_source / f"fold_{args.fold}_val.txt"

    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    image_by_name = {path.name: path for path in sorted(train_images.glob("*.png"))}
    mask_by_name = {path.name: path for path in sorted(train_masks.glob("*.png"))}
    if len(image_by_name) != 6996 or set(image_by_name) != set(mask_by_name):
        raise ValueError("Training images and masks are incomplete or do not match")

    train_names = read_split_list(train_split)
    val_names = read_split_list(val_split)
    overlap = sorted(set(train_names) & set(val_names))
    if overlap:
        raise ValueError(f"Train/val split overlap: {overlap[:5]}")
    expected = set(image_by_name)
    actual = set(train_names) | set(val_names)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Split files do not match official training files. "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    split_dir = output / "splits"
    split_dir.mkdir(parents=True)
    for path in sorted(split_source.glob("*")):
        if path.is_file():
            shutil.copy2(path, split_dir / path.name)

    for subset, names in (("train", train_names), ("val", val_names)):
        for name in names:
            materialize_file(
                image_by_name[name], output / subset / "images" / name, args.mode
            )
            materialize_file(
                mask_by_name[name], output / subset / "masks" / name, args.mode
            )

    for path in sorted(test_images.glob("*.png")):
        materialize_file(path, output / "test" / "images" / path.name, args.mode)

    summary = {
        "split_dir": str(split_source),
        "validation_fold": args.fold,
        "storage": args.mode,
        "hardlink_fallback": "copy",
        "train_images": len(train_names),
        "val_images": len(val_names),
        "test_images": len(list(test_images.glob("*.png"))),
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Dataset split\n\n"
        "The default split uses fold 0 for validation and folds 1-4 for training.\n\n"
        "- `train/`: 80% training images and masks\n"
        "- `val/`: 20% validation images and masks\n"
        "- `test/`: official test_1 images; never use these for training\n"
        "- `splits/`: reproducible manifests for all five folds\n\n"
        "The split is loaded from fixed txt files in `data_splits/`. "
        "Hard links are used by default when supported; otherwise files are copied. "
        "Do not treat Examples as extra data; those files duplicate training samples 0000-0010.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
