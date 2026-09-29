from __future__ import annotations

import csv
import json
import os
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


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    os.link(source, destination)


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
    workspace = Path(__file__).resolve().parent
    source = workspace / "2026-低空图像语义分割赛道-训练集"
    train_images = source / "train" / "train" / "images"
    train_masks = source / "train" / "train" / "masks"
    test_images = source / "test_1" / "images"
    output = workspace / "data"

    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    image_paths = sorted(train_images.glob("*.png"))
    mask_paths = sorted(train_masks.glob("*.png"))
    image_by_stem = {path.stem: path for path in image_paths}
    mask_by_stem = {path.stem: path for path in mask_paths}
    stems = sorted(image_by_stem.keys())
    if len(stems) != 6996 or set(stems) != set(mask_by_stem):
        raise ValueError("Training images and masks are incomplete or do not match")

    counts = read_mask_counts([mask_by_stem[stem] for stem in stems])
    features = build_stratification_features(counts)
    assignment = choose_folds(counts, features)

    split_dir = output / "splits"
    split_dir.mkdir(parents=True)
    with (split_dir / "folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "fold"])
        writer.writerows((f"{stem}.png", int(fold)) for stem, fold in zip(stems, assignment))

    for fold in range(N_FOLDS):
        val_names = [f"{stem}.png" for stem, value in zip(stems, assignment) if value == fold]
        train_names = [f"{stem}.png" for stem, value in zip(stems, assignment) if value != fold]
        (split_dir / f"fold_{fold}_train.txt").write_text(
            "\n".join(train_names) + "\n", encoding="utf-8"
        )
        (split_dir / f"fold_{fold}_val.txt").write_text(
            "\n".join(val_names) + "\n", encoding="utf-8"
        )

    # Materialize fold 0 as the default development split using space-saving hard links.
    for stem, fold in zip(stems, assignment):
        subset = "val" if fold == 0 else "train"
        hardlink(image_by_stem[stem], output / subset / "images" / f"{stem}.png")
        hardlink(mask_by_stem[stem], output / subset / "masks" / f"{stem}.png")

    for path in sorted(test_images.glob("*.png")):
        hardlink(path, output / "test" / "images" / path.name)

    summary = {
        "seed": SEED,
        "folds": N_FOLDS,
        "default_validation_fold": 0,
        "storage": "hard_links",
        "fold_summary": summarize(counts, assignment),
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
        "Files under train, val, and test are hard links to the original extracted dataset, "
        "so they consume almost no additional disk space. Do not treat Examples as extra data; "
        "those files duplicate training samples 0000-0010.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
