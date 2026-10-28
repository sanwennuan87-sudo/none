import numpy as np
import pandas as pd


def extract_audit_governance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute audit and corporate governance predictor variables defined in Table 5 of Bertomeu et al. (2020)."""
    out = df.sort_values(["Stkcd", "fyear"]).copy()

    total_fee_raw = out["audit_fee_raw"] + out["non_audit_fee_raw"]
    out["feeratio"] = out["non_audit_fee_raw"] / total_fee_raw.replace(0, np.nan)
    out["non_audit_fees"] = np.log(out["non_audit_fee_raw"].clip(lower=1.0))
    out["audit_fees"] = np.log(out["audit_fee_raw"].clip(lower=1.0))
    out["total_fee"] = np.log(total_fee_raw.clip(lower=1.0))

    # Percentile ranks by auditor and year
    out["ranknon"] = out.groupby(["auditor_id", "fyear"])["non_audit_fee_raw"].rank(pct=True)
    out["rankaud"] = out.groupby(["auditor_id", "fyear"])["audit_fee_raw"].rank(pct=True)
    out["ranktot"] = out.groupby(["auditor_id", "fyear"])["total_fee"].rank(pct=True)

    out["audit_tenure"] = out["auditor_tenure"]
    out["audit_ch"] = out["auditor_change"]

    # Map Table 5 exact abbreviations for governance variables
    out["AC_Affiliate"] = out["ac_affiliate"]
    out["AC_Chair_Affiliate"] = out["ac_chair_affiliate"]
    out["AC_Meetings"] = out["ac_meetings"]
    out["ACsize"] = out["ac_size"]
    out["AffiliateOwn"] = out["affiliate_own"]
    out["Appointed_Affiliate"] = out["appointed_affiliate"]
    out["Appointed_Outside"] = out["appointed_outside"]
    out["Board_Inside"] = out["board_inside"]
    out["Board_Meetings"] = out["board_meetings"]
    out["Boardsize"] = out["board_size"]
    out["Busy_Affiliate"] = out["busy_affiliate"]
    out["Busy_Insider"] = out["busy_insider"]
    out["Busy_Outsider"] = out["busy_outsider"]
    out["CC_Affiliate"] = out["cc_affiliate"]
    out["CC_Chair_Affiliate"] = out["cc_chair_affiliate"]
    out["CC_Meetings"] = out["cc_meetings"]
    out["CCsize"] = out["cc_size"]
    out["Insider_Chairman"] = out["insider_chairman"]
    out["Lead_Director"] = out["lead_director"]
    out["Outsider_Own"] = out["outsider_own"]

    return out
