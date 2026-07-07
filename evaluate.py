"""Evaluate the Roboflow casting-defect classifier on the held-out test set.

Usage:
    python evaluate.py            # full evaluation over ./test-set
    python evaluate.py --probe    # run ONE image and print the raw response JSON

Reads ROBOFLOW_API_KEY from .env. The key is never printed.
"""

import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv

TEST_DIR = Path(__file__).parent / "test-set"
POSITIVE_CLASS = "defect"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_ID = "dhruv-kothari-yrwsq/casting-defect-tcdc-demo-1-resnet18-t1"
INFERENCE_URL = f"https://serverless.roboflow.com/{MODEL_ID}"


def get_api_key():
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "PASTE_KEY_HERE":
        sys.exit("ROBOFLOW_API_KEY is not set in .env")
    return api_key


def run_model(api_key: str, image_path: str):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")
    resp = requests.post(
        INFERENCE_URL,
        params={"api_key": api_key},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()
    if "top" not in result:
        raise ValueError(f"Unexpected response: {json.dumps(result)[:500]}")
    return result["top"], float(result.get("confidence", float("nan")))


def collect_images():
    if not TEST_DIR.is_dir():
        sys.exit(f"Test directory not found: {TEST_DIR}")
    samples = []
    classes = sorted(d.name for d in TEST_DIR.iterdir() if d.is_dir())
    for class_name in classes:
        for path in sorted((TEST_DIR / class_name).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_name))
    return samples, classes


def probe(api_key: str):
    samples, _ = collect_images()
    image_path, true_label = samples[0]
    print(f"Probing with: {image_path.name} (true class: {true_label})")
    predicted, confidence = run_model(api_key, str(image_path))
    print(f"Result: top={predicted!r}, confidence={confidence:.4f}")


def save_confusion_matrix(matrix, classes, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), [c.capitalize() for c in classes])
    ax.set_yticks(range(len(classes)), [c.capitalize() for c in classes])
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title("Casting Defect Classifier — Confusion Matrix", fontsize=13, pad=14)
    threshold = matrix.max() / 2 if matrix.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate(api_key: str):
    samples, classes = collect_images()
    print(f"Found {len(samples)} images across classes: {', '.join(classes)}\n")

    rows = []
    failed = []
    for index, (image_path, true_label) in enumerate(samples, start=1):
        try:
            predicted, confidence = run_model(api_key, str(image_path))
            predicted = predicted.strip().lower()
        except Exception as exc:  # noqa: BLE001 — skip and report any failed image
            failed.append((image_path.name, str(exc)[:200]))
            rows.append({"filename": image_path.name, "true": true_label,
                         "predicted": "ERROR", "confidence": "", "correct": ""})
            continue
        rows.append({
            "filename": image_path.name,
            "true": true_label,
            "predicted": predicted,
            "confidence": f"{confidence:.4f}",
            "correct": predicted == true_label,
        })
        if index % 10 == 0:
            done = sum(1 for r in rows if r["correct"] is True)
            print(f"  {index}/{len(samples)} images processed "
                  f"({done} correct so far, {len(failed)} failed)")

    scored = [r for r in rows if r["predicted"] != "ERROR"]
    if not scored:
        sys.exit("\nNo images were successfully classified — cannot compute metrics.")

    # Overall and per-class accuracy
    correct = sum(r["correct"] for r in scored)
    print(f"\n{'=' * 50}")
    print(f"Overall accuracy: {correct}/{len(scored)} = {correct / len(scored):.2%}")
    for class_name in classes:
        class_rows = [r for r in scored if r["true"] == class_name]
        if class_rows:
            class_correct = sum(r["correct"] for r in class_rows)
            print(f"  {class_name:>8} accuracy: {class_correct}/{len(class_rows)} "
                  f"= {class_correct / len(class_rows):.2%}")

    # Binary counts, "defect" as positive
    tp = sum(1 for r in scored if r["true"] == POSITIVE_CLASS and r["predicted"] == POSITIVE_CLASS)
    fn = sum(1 for r in scored if r["true"] == POSITIVE_CLASS and r["predicted"] != POSITIVE_CLASS)
    fp = sum(1 for r in scored if r["true"] != POSITIVE_CLASS and r["predicted"] == POSITIVE_CLASS)
    tn = sum(1 for r in scored if r["true"] != POSITIVE_CLASS and r["predicted"] != POSITIVE_CLASS)
    print(f"\nWith '{POSITIVE_CLASS}' as the positive class:")
    print(f"  True positives:  {tp}")
    print(f"  False positives: {fp}")
    print(f"  True negatives:  {tn}")
    print(f"  False negatives: {fn}")

    # Confusion matrix PNG (rows = true, columns = predicted)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_index = {c: i for i, c in enumerate(classes)}
    for r in scored:
        if r["predicted"] in class_index:
            matrix[class_index[r["true"]], class_index[r["predicted"]]] += 1
    save_confusion_matrix(matrix, classes, Path(__file__).parent / "confusion_matrix.png")
    print("\nSaved confusion_matrix.png")

    # Per-image CSV
    csv_path = Path(__file__).parent / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true", "predicted", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path.name}")

    misclassified = [r for r in scored if not r["correct"]]
    if misclassified:
        print(f"\nMisclassified images ({len(misclassified)}):")
        for r in misclassified:
            print(f"  {r['filename']}  (true: {r['true']}, predicted: {r['predicted']}, "
                  f"confidence: {r['confidence']})")
    else:
        print("\nNo misclassified images.")

    if failed:
        print(f"\nImages that failed and were skipped ({len(failed)}):")
        for name, error in failed:
            print(f"  {name}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="Run one image and print the raw response, then exit.")
    args = parser.parse_args()
    api_key = get_api_key()
    if args.probe:
        probe(api_key)
    else:
        evaluate(api_key)


if __name__ == "__main__":
    main()
