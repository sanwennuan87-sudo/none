import json
import os
import pandas as pd
from src.export_cleaned_csv import export_cleaned_csv


def split_train_val_test(cleaned_df: pd.DataFrame, config: dict) -> dict:
    """Split cleaned panel data into Training (2001-2009), Validation (2010-2011), Combined Train+Val (2001-2011), and Test (2012-2014) sets per Bertomeu et al. (2020) Figure 1."""
    split_cfg = config.get("split_settings", {})
    train_start, train_end = split_cfg.get("train_years", [2001, 2009])
    val_start, val_end = split_cfg.get("validation_years", [2010, 2011])
    comb_start, comb_end = split_cfg.get("train_val_combined_years", [2001, 2011])
    test_start, test_end = split_cfg.get("test_years", [2012, 2014])

    out_paths = split_cfg.get("output_paths", {})
    train_path = out_paths.get("train_csv", "data/train_sample_2001_2009.csv")
    val_path = out_paths.get("validation_csv", "data/validation_sample_2010_2011.csv")
    comb_path = out_paths.get("train_val_combined_csv", "data/train_val_combined_2001_2011.csv")
    test_path = out_paths.get("test_csv", "data/test_sample_2012_2014.csv")
    report_path = out_paths.get("split_summary_json", "data/split_summary_report.json")

    train_df = cleaned_df.loc[cleaned_df["fyear"].between(train_start, train_end)].reset_index(drop=True)
    val_df = cleaned_df.loc[cleaned_df["fyear"].between(val_start, val_end)].reset_index(drop=True)
    train_val_df = cleaned_df.loc[cleaned_df["fyear"].between(comb_start, comb_end)].reset_index(drop=True)
    test_df = cleaned_df.loc[cleaned_df["fyear"].between(test_start, test_end)].reset_index(drop=True)

    saved_train = export_cleaned_csv(train_df, train_path)
    saved_val = export_cleaned_csv(val_df, val_path)
    saved_comb = export_cleaned_csv(train_val_df, comb_path)
    saved_test = export_cleaned_csv(test_df, test_path)

    feature_cols = [c for c in cleaned_df.columns if c not in ("Stkcd", "fyear", "misstatement")]
    split_report = {
        "reference": "Bertomeu et al. (2020), Section 2 & Figure 1",
        "schema": {
            "id_columns": ["Stkcd", "fyear"],
            "target_column": "misstatement",
            "predictor_feature_count": len(feature_cols),
            "predictor_columns": feature_cols
        },
        "splits": {
            "train_2001_2009": {
                "years": f"{train_start}-{train_end}",
                "rows": len(train_df),
                "misstatement_count": int(train_df["misstatement"].sum()),
                "csv_path": saved_train
            },
            "validation_2010_2011": {
                "years": f"{val_start}-{val_end}",
                "rows": len(val_df),
                "misstatement_count": int(val_df["misstatement"].sum()),
                "csv_path": saved_val
            },
            "train_val_combined_2001_2011": {
                "years": f"{comb_start}-{comb_end}",
                "rows": len(train_val_df),
                "misstatement_count": int(train_val_df["misstatement"].sum()),
                "csv_path": saved_comb
            },
            "test_2012_2014": {
                "years": f"{test_start}-{test_end}",
                "rows": len(test_df),
                "misstatement_count": int(test_df["misstatement"].sum()),
                "csv_path": saved_test
            }
        }
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=4, ensure_ascii=False)

    return split_report
