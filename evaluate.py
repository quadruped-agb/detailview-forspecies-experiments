"""
Evaluate DetailView predictions against ground truth metadata.
"""

import argparse
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to DetailView predictions CSV")
    parser.add_argument("--truth", required=True, help="Path to ground truth metadata CSV")
    parser.add_argument("--tree_id_col", default="TreeID", help="Column name for tree ID in both files (after normalization)")
    parser.add_argument("--pred_species_col", default="species", help="Predicted species column name in predictions CSV")
    parser.add_argument("--truth_species_col", default="species_full", help="Actual species column name in ground truth CSV (after code mapping)")
    parser.add_argument("--out_dir", default="results", help="Where to save outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    preds = pd.read_csv(args.pred)
    truth = pd.read_csv(args.truth)

    if "filename" in preds.columns:
        preds["TreeID"] = (
            preds["filename"].str.split("/").str[-1].str.replace(".las", "", regex=False)
        )

    SPECIES_CODE_MAP = {
        "QUEFAG": "Quercus_faginea",
        "PINNIG": "Pinus_nigra",
        "QUEILE": "Quercus_ilex",
        "PINSYL": "Pinus_sylvestris",
        "PINPIN": "Pinus_pinaster",
    }
    EXCLUDE_CODES = {"QUERCUS", "JUNIPE", "DEAD"}

    if "sp" in truth.columns:
        n_before = len(truth)
        truth = truth[~truth["sp"].isin(EXCLUDE_CODES)].copy()
        n_excluded = n_before - len(truth)
        if n_excluded:
            print(f"Excluded {n_excluded} ground truth rows with genus-only/invalid codes: {EXCLUDE_CODES}")
        truth["species_full"] = truth["sp"].map(SPECIES_CODE_MAP)
        unmapped = truth[truth["species_full"].isna()]["sp"].unique()
        if len(unmapped):
            print(f"Warning: unmapped species codes found, add to SPECIES_CODE_MAP: {list(unmapped)}")
        truth = truth.rename(columns={"id": "TreeID"})

    merged = preds.merge(
        truth, on=args.tree_id_col, how="inner", suffixes=("_pred", "_truth")
    )

    if merged.empty:
        raise ValueError(
            "No matching TreeIDs between predictions and ground truth. "
            "Check that the tree_id_col values match exactly in both files."
        )

    n_pred = len(preds)
    n_truth = len(truth)
    n_matched = len(merged)
    if n_matched < n_pred or n_matched < n_truth:
        print(
            f"Warning: only {n_matched} trees matched "
            f"({n_pred} in predictions, {n_truth} in ground truth). "
            "Unmatched trees are excluded from metrics."
        )

    y_true = merged[args.truth_species_col]
    y_pred = merged[args.pred_species_col]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy: {acc:.3f} ({n_matched} trees)\n")

    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Overall accuracy: {acc:.3f} ({n_matched} trees)\n\n")
        f.write(report)

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    plt.figure(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.title(f"DetailView confusion matrix (n={n_matched})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=200)
    print(f"\nSaved: {args.out_dir}/classification_report.txt")
    print(f"Saved: {args.out_dir}/confusion_matrix.csv")
    print(f"Saved: {args.out_dir}/confusion_matrix.png")

    merged["correct"] = y_true == y_pred
    merged.to_csv(os.path.join(args.out_dir, "merged_results.csv"), index=False)
    print(f"Saved: {args.out_dir}/merged_results.csv")


if __name__ == "__main__":
    main()
