from src.load_config import load_config
from src.verify_csmar_api import verify_csmar_api
from src.verify_school_cas_login import verify_school_cas_login
from src.fetch_csmar_raw_data import fetch_csmar_raw_data
from src.clean_sample_selection import clean_sample_selection
from src.extract_accounting_features import extract_accounting_features
from src.extract_audit_governance_features import extract_audit_governance_features
from src.extract_market_business_features import extract_market_business_features
from src.clean_and_winsorize_features import clean_and_winsorize_features
from src.export_cleaned_csv import export_cleaned_csv
from src.split_train_val_test import split_train_val_test


def run_pipeline(config_path: str = "config.json") -> dict:
    """Orchestrate API verification, CSMAR data extraction, 102-predictor feature cleaning, and Figure 1 Train/Validation/Test chronological splitting."""
    config = load_config(config_path)
    api_report = verify_csmar_api(config)
    cas_report = verify_school_cas_login(config)

    raw_df = fetch_csmar_raw_data(config, api_report)
    acc_df = extract_accounting_features(raw_df)
    aud_gov_df = extract_audit_governance_features(acc_df)
    full_feat_df = extract_market_business_features(aud_gov_df)

    pipe_cfg = config.get("pipeline_settings", {})
    s_start = int(pipe_cfg.get("sample_start_year", 2001))
    s_end = int(pipe_cfg.get("sample_end_year", 2014))
    selected_df = clean_sample_selection(full_feat_df, start_year=s_start, end_year=s_end)
    cleaned_df = clean_and_winsorize_features(selected_df, config)

    out_csv = pipe_cfg.get("output_csv_path", "data/cleaned_misstatement_features.csv")
    saved_full_path = export_cleaned_csv(cleaned_df, out_csv)

    split_report = split_train_val_test(cleaned_df, config)

    summary = {
        "can_institutional_account_call_api_directly": api_report["can_institutional_account_call_python_api_directly"],
        "cas_status": cas_report["session_status"],
        "total_columns_per_csv": len(cleaned_df.columns),
        "predictor_feature_count_X": split_report["schema"]["predictor_feature_count"],
        "target_column_y": split_report["schema"]["target_column"],
        "train_2001_2009_rows": split_report["splits"]["train_2001_2009"]["rows"],
        "validation_2010_2011_rows": split_report["splits"]["validation_2010_2011"]["rows"],
        "train_val_combined_2001_2011_rows": split_report["splits"]["train_val_combined_2001_2011"]["rows"],
        "test_2012_2014_rows": split_report["splits"]["test_2012_2014"]["rows"],
        "full_csv_path": saved_full_path
    }
    print("=== Pipeline Execution & Dataset Split Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    run_pipeline()
