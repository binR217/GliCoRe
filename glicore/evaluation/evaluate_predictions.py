"""Evaluate GliCoRe predictions with the metrics reported in the paper."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import SimpleITK as sitk


REGIONS = {
    "brats": {"WT": (1, 2, 3), "ET": (3,), "TC": (2, 3)},
    "synapse": {
        "Spl": (1,), "RKid": (2,), "LKid": (3,), "Gal": (4,),
        "Liv": (6,), "Sto": (7,), "Aor": (8,), "Pan": (11,),
    },
    "acdc": {"RV": (1,), "Myo": (2,), "LV": (3,)},
}


def _strip_nii_suffix(path):
    name = Path(path).name
    if not name.endswith(".nii.gz"):
        raise ValueError("Expected a .nii.gz file: %s" % path)
    return name[:-7]


def canonical_case_id(path, dataset):
    case_id = _strip_nii_suffix(path)
    if dataset == "synapse" and case_id.startswith("label"):
        return "img" + case_id[5:]
    if dataset == "acdc" and case_id.endswith("_gt"):
        return case_id[:-3]
    return case_id


def read_nifti(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_zyx = tuple(float(value) for value in image.GetSpacing()[::-1])
    return array, spacing_zyx


def dice_score(prediction, target):
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def surface_distances(prediction, target, spacing):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return np.empty(0), np.empty(0)
    if not prediction.any() or not target.any():
        raise ValueError("Surface distances are undefined when only one mask is empty")

    structure = ndimage.generate_binary_structure(prediction.ndim, 1)
    pred_surface = np.logical_xor(
        prediction, ndimage.binary_erosion(prediction, structure=structure)
    )
    target_surface = np.logical_xor(
        target, ndimage.binary_erosion(target, structure=structure)
    )
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface, sampling=spacing
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        ~pred_surface, sampling=spacing
    )
    return distance_to_target[pred_surface], distance_to_prediction[target_surface]


def surface_metrics(prediction, target, spacing, nsd_tolerance_mm):
    pred_to_target, target_to_pred = surface_distances(prediction, target, spacing)
    if pred_to_target.size == 0 and target_to_pred.size == 0:
        return 0.0, 1.0, 0.0
    distances = np.concatenate((pred_to_target, target_to_pred))
    return (
        float(np.percentile(distances, 95)),
        float(np.mean(distances <= nsd_tolerance_mm)),
        float(distances.mean()),
    )


def index_directory(directory, dataset):
    indexed = {}
    for path in sorted(Path(directory).glob("*.nii.gz")):
        case_id = canonical_case_id(path, dataset)
        if case_id in indexed:
            raise RuntimeError("Duplicate case ID %s in %s" % (case_id, directory))
        indexed[case_id] = path
    if not indexed:
        raise RuntimeError("No .nii.gz files found in %s" % directory)
    return indexed


def evaluate_case(pred_path, gt_path, region_map, nsd_tolerance_mm):
    prediction, pred_spacing = read_nifti(pred_path)
    target, target_spacing = read_nifti(gt_path)
    if prediction.shape != target.shape:
        raise ValueError("Shape mismatch: %s versus %s" % (pred_path, gt_path))
    if not np.allclose(pred_spacing, target_spacing, rtol=0.0, atol=1e-4):
        raise ValueError("Spacing mismatch: %s versus %s" % (pred_path, gt_path))

    result = {}
    for name, label_ids in region_map.items():
        pred_mask = np.isin(prediction, label_ids)
        target_mask = np.isin(target, label_ids)
        hd95, nsd, masd = surface_metrics(
            pred_mask, target_mask, target_spacing, nsd_tolerance_mm
        )
        result[name] = {
            "DSC": 100.0 * dice_score(pred_mask, target_mask),
            "HD95": hd95,
            "NSD": 100.0 * nsd,
            "MASD": masd,
        }
    return result


def evaluate_directory(dataset, pred_dir, gt_dir, nsd_tolerance_mm):
    predictions = index_directory(pred_dir, dataset)
    targets = index_directory(gt_dir, dataset)
    if set(predictions) != set(targets):
        missing = sorted(set(targets) - set(predictions))
        extra = sorted(set(predictions) - set(targets))
        raise RuntimeError("Case mismatch; missing=%s extra=%s" % (missing, extra))

    rows = []
    for case_id in sorted(predictions):
        metrics = evaluate_case(
            predictions[case_id], targets[case_id], REGIONS[dataset],
            nsd_tolerance_mm,
        )
        row = {"case_id": case_id}
        for region, values in metrics.items():
            for metric, value in values.items():
                row["%s_%s" % (region, metric)] = value
        rows.append(row)
    return rows


def summarize(rows, dataset):
    regions = tuple(REGIONS[dataset])
    metrics = ("DSC", "HD95", "NSD", "MASD")
    summary = {"num_cases": len(rows), "regions": {}}
    for region in regions:
        summary["regions"][region] = {
            metric: float(np.mean([
                row["%s_%s" % (region, metric)] for row in rows
            ]))
            for metric in metrics
        }
    summary["average"] = {
        metric: float(np.mean([
            summary["regions"][region][metric] for region in regions
        ]))
        for metric in metrics
    }
    return summary


def save(rows, summary, output_prefix):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=tuple(REGIONS))
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--nsd-tolerance-mm", required=True, type=float)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rows = evaluate_directory(
        args.dataset, args.pred_dir, args.gt_dir, args.nsd_tolerance_mm
    )
    summary = summarize(rows, args.dataset)
    csv_path, json_path = save(rows, summary, args.output_prefix)
    print(json.dumps(summary, indent=2))
    print("Case metrics:", csv_path)
    print("Summary:", json_path)


if __name__ == "__main__":
    main()
