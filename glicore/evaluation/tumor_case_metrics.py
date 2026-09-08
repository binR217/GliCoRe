import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


BRATS_REGIONS = {
    "WT": (1, 2, 3),
    "TC": (2, 3),
    "ET": (3,),
}


def case_id_from_path(path):
    name = Path(path).name
    if not name.endswith(".nii.gz"):
        raise ValueError("Expected a .nii.gz file, got %s" % name)
    return name[:-7]


def regions_from_labels(labels, region_definitions=BRATS_REGIONS):
    labels = np.asarray(labels)
    return {
        name: np.isin(labels, class_ids)
        for name, class_ids in region_definitions.items()
    }


def dice_score(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / denom)


def hd95_score(pred, target, spacing):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    pred_nonempty = bool(pred.any())
    target_nonempty = bool(target.any())
    if not pred_nonempty and not target_nonempty:
        return 0.0
    if pred_nonempty != target_nonempty:
        extent = np.asarray(pred.shape, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
        return float(np.linalg.norm(extent))

    from medpy.metric.binary import hd95

    return float(hd95(pred, target, voxelspacing=spacing, connectivity=1))


def _read_nifti(path):
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_zyx = tuple(float(v) for v in image.GetSpacing()[::-1])
    return array, spacing_zyx


def evaluate_case(pred_path, gt_path):
    pred, pred_spacing = _read_nifti(pred_path)
    target, target_spacing = _read_nifti(gt_path)
    if pred.shape != target.shape:
        raise ValueError("Shape mismatch for %s: prediction %s, GT %s" %
                         (case_id_from_path(pred_path), pred.shape, target.shape))
    if not np.allclose(pred_spacing, target_spacing, rtol=0.0, atol=1e-4):
        raise ValueError("Spacing mismatch for %s: prediction %s, GT %s" %
                         (case_id_from_path(pred_path), pred_spacing, target_spacing))
    if not set(np.unique(pred)).issubset({0, 1, 2, 3}):
        raise ValueError("Prediction contains labels outside {0,1,2,3}: %s" % pred_path)

    pred_regions = regions_from_labels(pred)
    target_regions = regions_from_labels(target)
    row = {"case_id": case_id_from_path(pred_path)}
    for region_name in BRATS_REGIONS:
        row["%s_dice" % region_name] = dice_score(pred_regions[region_name], target_regions[region_name])
        row["%s_hd95" % region_name] = hd95_score(
            pred_regions[region_name], target_regions[region_name], target_spacing
        )
    return row


def _summary(rows):
    summary = {"num_cases": len(rows), "regions": {}}
    for region in BRATS_REGIONS:
        dice_values = np.asarray([row["%s_dice" % region] for row in rows], dtype=np.float64)
        hd_values = np.asarray([row["%s_hd95" % region] for row in rows], dtype=np.float64)
        worst_count = min(5, len(rows))
        summary["regions"][region] = {
            "dice_mean": float(dice_values.mean()),
            "dice_p10": float(np.percentile(dice_values, 10)),
            "hd95_mean": float(hd_values.mean()),
            "hd95_p90": float(np.percentile(hd_values, 90)),
            "hd95_max": float(hd_values.max()),
            "hd95_worst5_mean": float(np.sort(hd_values)[-worst_count:].mean()),
        }
    summary["average"] = {
        "dice_mean": float(np.mean([summary["regions"][r]["dice_mean"] for r in BRATS_REGIONS])),
        "hd95_mean": float(np.mean([summary["regions"][r]["hd95_mean"] for r in BRATS_REGIONS])),
    }
    return summary


def load_case_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        case_id = row.pop("case_id")
        result[case_id] = {key: float(value) for key, value in row.items()}
    return result


def paired_summary(rows, reference_csv):
    reference = load_case_csv(reference_csv)
    current = {row["case_id"]: row for row in rows}
    if set(reference) != set(current):
        raise RuntimeError(
            "Reference/current case IDs differ. missing=%s extra=%s" %
            (sorted(set(reference) - set(current)), sorted(set(current) - set(reference))))
    result = {"reference_csv": str(reference_csv), "regions": {}}
    for region in BRATS_REGIONS:
        dice_delta = np.asarray([
            current[case_id]["%s_dice" % region] - reference[case_id]["%s_dice" % region]
            for case_id in sorted(current)
        ])
        hd_improvement = np.asarray([
            reference[case_id]["%s_hd95" % region] - current[case_id]["%s_hd95" % region]
            for case_id in sorted(current)
        ])
        result["regions"][region] = {
            "dice_delta_mean": float(dice_delta.mean()),
            "dice_improved_fraction": float((dice_delta > 1e-8).mean()),
            "dice_regressed_fraction": float((dice_delta < -1e-8).mean()),
            "hd95_improvement_mean": float(hd_improvement.mean()),
            "hd95_improved_fraction": float((hd_improvement > 1e-8).mean()),
            "hd95_regressed_fraction": float((hd_improvement < -1e-8).mean()),
        }
    return result


def evaluate_directory(pred_dir, gt_dir, expected_count=None):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if expected_count is not None and len(pred_files) != expected_count:
        raise RuntimeError("Expected %d predictions, found %d in %s" %
                           (expected_count, len(pred_files), pred_dir))
    if not pred_files:
        raise RuntimeError("No .nii.gz predictions found in %s" % pred_dir)

    print(
        "Found %d predictions. Computing WT/TC/ET Dice and HD95..." % len(pred_files),
        flush=True,
    )

    rows = []
    seen = set()
    total_start = time.time()
    for index, pred_path in enumerate(pred_files, start=1):
        case_id = case_id_from_path(pred_path)
        if case_id in seen:
            raise RuntimeError("Duplicate prediction case ID: %s" % case_id)
        seen.add(case_id)
        gt_path = gt_dir / (case_id + ".nii.gz")
        if not gt_path.is_file():
            raise FileNotFoundError("Missing GT for %s: %s" % (case_id, gt_path))
        case_start = time.time()
        print("[%d/%d] %s ..." % (index, len(pred_files), case_id), end=" ", flush=True)
        rows.append(evaluate_case(pred_path, gt_path))
        print("done (%.1fs)" % (time.time() - case_start), flush=True)
    print("All cases completed in %.1fs." % (time.time() - total_start), flush=True)
    return rows, _summary(rows)


def save_results(rows, summary, out_prefix):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    fieldnames = ["case_id"]
    for region in BRATS_REGIONS:
        fieldnames.extend(["%s_dice" % region, "%s_hd95" % region])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Compute case-level BraTS WT/TC/ET Dice and HD95.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--reference-csv")
    args = parser.parse_args()

    rows, summary = evaluate_directory(args.pred_dir, args.gt_dir, args.expected_count)
    if args.reference_csv:
        summary["paired_vs_reference"] = paired_summary(rows, args.reference_csv)
    csv_path, json_path = save_results(rows, summary, args.out_prefix)
    print(json.dumps(summary, indent=2))
    print("case metrics:", csv_path)
    print("summary:", json_path)


if __name__ == "__main__":
    main()
