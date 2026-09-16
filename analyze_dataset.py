from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a semantic-segmentation dataset.")
    parser.add_argument("root", type=Path, help="Directory containing images/ and masks/")
    args = parser.parse_args()

    image_paths = sorted((args.root / "images").glob("*.png"))
    mask_paths = sorted((args.root / "masks").glob("*.png"))
    image_by_stem = {path.stem: path for path in image_paths}
    mask_by_stem = {path.stem: path for path in mask_paths}
    common_stems = sorted(image_by_stem.keys() & mask_by_stem.keys())

    image_dimensions: collections.Counter[tuple[int, int]] = collections.Counter()
    mask_dimensions: collections.Counter[tuple[int, int]] = collections.Counter()
    image_modes: collections.Counter[str] = collections.Counter()
    mask_modes: collections.Counter[str] = collections.Counter()
    pixel_counts = np.zeros(256, dtype=np.int64)
    image_presence = np.zeros(256, dtype=np.int64)
    per_image_counts: list[np.ndarray] = []
    size_mismatches: list[dict[str, object]] = []

    for stem in common_stems:
        with Image.open(image_by_stem[stem]) as image:
            image_dimensions[image.size] += 1
            image_modes[image.mode] += 1
            image_size = image.size
        with Image.open(mask_by_stem[stem]) as mask:
            mask_dimensions[mask.size] += 1
            mask_modes[mask.mode] += 1
            mask_size = mask.size
            values = np.asarray(mask)

        if image_size != mask_size:
            size_mismatches.append(
                {"stem": stem, "image_size": image_size, "mask_size": mask_size}
            )
        counts = np.bincount(values.reshape(-1), minlength=256)
        pixel_counts += counts
        image_presence[np.flatnonzero(counts)] += 1
        per_image_counts.append(counts)

    total_pixels = int(pixel_counts.sum())
    per_image = np.stack(per_image_counts)
    class_distribution = {}
    for label in np.flatnonzero(pixel_counts):
        positive = per_image[:, label][per_image[:, label] > 0]
        descending = np.sort(per_image[:, label])[::-1]
        top_20_count = max(1, round(len(common_stems) * 0.2))
        class_distribution[str(label)] = {
            "positive_image_pixel_quantiles": {
                str(q): int(np.quantile(positive, q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "pixel_share_in_top_20pct_images": round(
                float(descending[:top_20_count].sum() / descending.sum() * 100), 3
            ),
        }

    normalized = per_image[:, :9] / np.maximum(per_image[:, :9].sum(axis=1, keepdims=True), 1)
    adjacent_distance = np.abs(normalized[1:] - normalized[:-1]).mean(axis=1)
    rng = np.random.default_rng(20260912)
    random_order = rng.permutation(len(normalized))
    random_distance = np.abs(
        normalized[random_order[1:]] - normalized[random_order[:-1]]
    ).mean(axis=1)

    report = {
        "pairing": {
            "images": len(image_paths),
            "masks": len(mask_paths),
            "paired": len(common_stems),
            "missing_masks": sorted(image_by_stem.keys() - mask_by_stem.keys()),
            "missing_images": sorted(mask_by_stem.keys() - image_by_stem.keys()),
        },
        "image_dimensions": {str(key): value for key, value in image_dimensions.items()},
        "image_modes": dict(image_modes),
        "mask_dimensions": {str(key): value for key, value in mask_dimensions.items()},
        "mask_modes": dict(mask_modes),
        "size_mismatches": size_mismatches,
        "classes": {
            str(label): {
                "pixels": int(pixel_counts[label]),
                "pixel_pct": round(float(pixel_counts[label] / total_pixels * 100), 6),
                "images_present": int(image_presence[label]),
                "image_pct": round(float(image_presence[label] / len(common_stems) * 100), 3),
            }
            for label in np.flatnonzero(pixel_counts)
        },
        "class_concentration": class_distribution,
        "sequence_check": {
            "adjacent_mean_l1_per_class": round(float(adjacent_distance.mean()), 6),
            "random_mean_l1_per_class": round(float(random_distance.mean()), 6),
            "adjacent_median_l1_per_class": round(float(np.median(adjacent_distance)), 6),
            "random_median_l1_per_class": round(float(np.median(random_distance)), 6),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
