import numpy as np
import pandas as pd


def clean_and_winsorize_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create Table 5 missing indicators, winsorize continuous variables at 1%/99% (fitted on training period to avoid look-ahead bias), impute missing values to 0, and return strictly [Stkcd, fyear, misstatement] + 102 predictors."""
    out = df.copy()
    pipe_cfg = config.get("pipeline_settings", {})
    split_cfg = config.get("split_settings", {})
    lower_q = float(pipe_cfg.get("winsorize_lower", 0.01))
    upper_q = float(pipe_cfg.get("winsorize_upper", 0.99))
    fit_train_only = bool(pipe_cfg.get("winsorize_fit_on_train_only", True))
    train_start, train_end = split_cfg.get("train_years", [2001, 2009])

    out["misstatement"] = pd.to_numeric(out["misstatement"], errors="coerce").fillna(0).astype(int)

    # Explicit Table 5 missing value indicators (Section 2: missing variables set to 0 with indicator = 1)
    out["missing_Board"] = out["Boardsize"].isna().astype(int)
    out["missing_DD"] = (out["resid"].isna() | out["sresid"].isna()).astype(int)
    out["missing_DA"] = (out["da"].isna() | out["dadif"].isna()).astype(int)
    out["missing_TAX"] = out["tax"].isna().astype(int)
    out["missing_OB"] = out["ch_backlog"].isna().astype(int)
    out["missing_Pension"] = (out["pension"].isna() | out["ch_pension"].isna()).astype(int)
    out["missing_AF"] = (out["af_error"].isna() | out["disper"].isna()).astype(int)
    out["missing_Rating"] = out["credit_rating"].isna().astype(int)
    out["missing_auopic"] = out["auopic"].isna().astype(int)
    out["missing_going_concern"] = out["going_concern"].isna().astype(int)

    continuous_cols = [
        "WC_acc", "rsst_ac", "ch_rec", "ch_inv", "soft_assets", "da", "dadif",
        "resid", "sresid", "ch_cs", "ch_cm", "ch_roa", "ch_fcf", "tax", "oplease",
        "pension", "ch_pension", "ch_emp", "ch_backlog", "cff", "leverage",
        "ret_t", "ret_t_1", "bm", "ep", "AC_Affiliate", "AC_Meetings", "ACsize",
        "AffiliateOwn", "Appointed_Affiliate", "Appointed_Outside", "Board_Inside",
        "Board_Meetings", "Boardsize", "Busy_Affiliate", "Busy_Insider", "Busy_Outsider",
        "CC_Affiliate", "CC_Meetings", "CCsize", "Outsider_Own", "feeratio",
        "ranknon", "rankaud", "ranktot", "non_audit_fees", "audit_fees", "total_fee",
        "age", "audit_tenure", "af_error", "disper", "spread", "vol", "short",
        "credit_rating", "rating_chg"
    ]

    train_mask = out["fyear"].between(train_start, train_end)
    for col in continuous_cols:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            ref_series = s.loc[train_mask].dropna() if (fit_train_only and train_mask.any()) else s.dropna()
            if len(ref_series) > 0:
                q_low = float(ref_series.quantile(lower_q))
                q_high = float(ref_series.quantile(upper_q))
                s = s.clip(lower=q_low, upper=q_high)
            out[col] = s.fillna(0.0).round(6)

    indicator_cols = [
        "leasedum", "issue", "exfin", "ind1", "ind2", "ind3", "ind4", "ind5",
        "ind6", "ind7", "ind8", "ind9", "ind10", "ind11", "ind12", "ind13",
        "ind14", "ind15", "AC_Chair_Affiliate", "CC_Chair_Affiliate",
        "Insider_Chairman", "Lead_Director", "big4", "audit_ch",
        "going_concern", "auop", "auopic", "beat", "mf", "foreign",
        "res_an0", "res_an1", "res_an2", "res_an3", "rating_downgrade",
        "missing_Board", "missing_DD", "missing_DA", "missing_TAX",
        "missing_OB", "missing_Pension", "missing_AF", "missing_Rating",
        "missing_auopic", "missing_going_concern"
    ]

    for col in indicator_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # Strictly keep ONLY: [Stkcd, fyear, misstatement] + 102 predictor features (X)
    ordered_cols = ["Stkcd", "fyear", "misstatement"] + continuous_cols + indicator_cols
    existing_ordered = [c for c in ordered_cols if c in out.columns]
    return out[existing_ordered]
