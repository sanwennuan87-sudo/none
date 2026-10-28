import numpy as np
import pandas as pd


def extract_accounting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accounting & accrual predictor variables defined in Table 5 of Bertomeu et al. (2020)."""
    out = df.sort_values(["Stkcd", "fyear"]).copy()
    g = out.groupby("Stkcd")

    avg_assets = 0.5 * (out["total_assets"] + g["total_assets"].shift(1))
    beg_assets = g["total_assets"].shift(1)

    d_ca = out["curr_assets"] - g["curr_assets"].shift(1)
    d_cash = out["cash_sti"] - g["cash_sti"].shift(1)
    d_cl = out["curr_liab"] - g["curr_liab"].shift(1)
    d_std = out["st_debt"] - g["st_debt"].shift(1)
    d_tp = out["taxes_payable"] - g["taxes_payable"].shift(1)
    d_rec = out["receivables"] - g["receivables"].shift(1)
    d_inv = out["inventory"] - g["inventory"].shift(1)
    d_ap = out["accounts_payable"] - g["accounts_payable"].shift(1)
    d_sales = out["sales"] - g["sales"].shift(1)
    d_ppe = out["ppe"] - g["ppe"].shift(1)

    # 1. WC accruals: [(dCA - dCash) - (dCL - dSTD - dTP)] / Average Total Assets
    out["WC_acc"] = ((d_ca - d_cash) - (d_cl - d_std - d_tp)) / avg_assets

    # 2. RSST accruals: (dWC + dNCO + dFIN) / Average Total Assets
    wc_level = (out["curr_assets"] - out["cash_sti"]) - (out["curr_liab"] - out["st_debt"])
    nco_level = (out["total_assets"] - out["curr_assets"] - out["lt_investments"]) - (
        out["total_liab"] - out["curr_liab"] - out["lt_debt"]
    )
    fin_level = (out["cash_sti"] + out["lt_investments"]) - (
        out["lt_debt"] + out["st_debt"] + out["pref_stock"]
    )
    out["_wc_level"] = wc_level
    out["_nco_level"] = nco_level
    out["_fin_level"] = fin_level
    g = out.groupby("Stkcd")
    d_wc = out["_wc_level"] - g["_wc_level"].shift(1)
    d_nco = out["_nco_level"] - g["_nco_level"].shift(1)
    d_fin = out["_fin_level"] - g["_fin_level"].shift(1)
    out["rsst_ac"] = (d_wc + d_nco + d_fin) / avg_assets

    # 3. Change in receivables & inventory
    out["ch_rec"] = d_rec / avg_assets
    out["ch_inv"] = d_inv / avg_assets

    # 4. % Soft assets: (Total Assets - PP&E - Cash and Cash Equivalent) / Total Assets
    out["soft_assets"] = (out["total_assets"] - out["ppe"] - out["cash_sti"]) / out["total_assets"]

    # 5. Cross-sectional Modified Jones discretionary accruals (da) & Performance-matched (dadif)
    out["sic2"] = (out["sic"] // 100).astype(int)
    out["_x1"] = 1.0 / beg_assets
    out["_x2"] = (d_sales - d_rec) / beg_assets
    out["_x3"] = d_ppe / beg_assets
    out["roa_t"] = out["earnings"] / avg_assets

    da_vals = np.full(len(out), np.nan)
    for (_, _), idx in out.groupby(["sic2", "fyear"]).groups.items():
        sub = out.loc[idx, ["WC_acc", "_x1", "_x2", "_x3"]].dropna()
        if len(sub) >= 5:
            X = np.column_stack([np.ones(len(sub)), sub["_x1"].values, sub["_x2"].values, sub["_x3"].values])
            y = sub["WC_acc"].values
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            da_vals[sub.index] = resid
    out["da"] = da_vals

    dadif_vals = np.full(len(out), np.nan)
    for (_, _), idx in out.groupby(["sic2", "fyear"]).groups.items():
        sub = out.loc[idx, ["da", "roa_t"]].dropna()
        if len(sub) >= 2:
            roas = sub["roa_t"].values
            das = sub["da"].values
            for i_pos, row_idx in enumerate(sub.index):
                diffs = np.abs(roas - roas[i_pos])
                diffs[i_pos] = np.inf
                match_pos = int(np.argmin(diffs))
                dadif_vals[row_idx] = das[i_pos] - das[match_pos]
    out["dadif"] = dadif_vals

    # 6. Dechow-Dichev (2002) lagged residuals: dWC = b0 + b1*CFO_{t-1} + b2*CFO_t + b3*CFO_{t+1} + e
    out["_d_wc_scaled"] = d_wc / avg_assets
    out["_cfo_scaled"] = out["cfo"] / avg_assets
    g = out.groupby("Stkcd")
    out["_cfo_lag1"] = g["_cfo_scaled"].shift(1)
    out["_cfo_lead1"] = g["_cfo_scaled"].shift(-1)

    dd_resid = np.full(len(out), np.nan)
    dd_sresid = np.full(len(out), np.nan)
    for _, idx in out.groupby("sic2").groups.items():
        sub = out.loc[idx, ["_d_wc_scaled", "_cfo_lag1", "_cfo_scaled", "_cfo_lead1"]].dropna()
        if len(sub) >= 6:
            X = np.column_stack([
                np.ones(len(sub)),
                sub["_cfo_lag1"].values,
                sub["_cfo_scaled"].values,
                sub["_cfo_lead1"].values
            ])
            y = sub["_d_wc_scaled"].values
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            abs_r = np.abs(r)
            mean_abs = float(np.mean(abs_r))
            std_r = max(1e-8, float(np.std(r, ddof=1)))
            dd_resid[sub.index] = abs_r - mean_abs
            dd_sresid[sub.index] = r / std_r
    out["_dd_resid_curr"] = dd_resid
    out["_dd_sresid_curr"] = dd_sresid
    g = out.groupby("Stkcd")
    # Paper Section 2: use lag 1-year value for DD residuals to avoid future look-ahead bias
    out["resid"] = g["_dd_resid_curr"].shift(1)
    out["sresid"] = g["_dd_sresid_curr"].shift(1)

    # 7. Cash sales, cash margin, ROA change, FCF change, tax, operating leases, pension, financing
    cash_sales = out["sales"] - d_rec
    out["_cash_sales"] = cash_sales
    cash_margin = 1.0 - ((out["cogs"] - d_inv + d_ap) / cash_sales.replace(0, np.nan))
    out["_cash_margin"] = cash_margin
    out["_fcf_raw"] = out["earnings"] - (d_wc + d_nco + d_fin)
    g = out.groupby("Stkcd")

    out["ch_cs"] = (out["_cash_sales"] - g["_cash_sales"].shift(1)) / g["_cash_sales"].shift(1).abs().replace(0, np.nan)
    out["ch_cm"] = (out["_cash_margin"] - g["_cash_margin"].shift(1)) / g["_cash_margin"].shift(1).abs().replace(0, np.nan)
    out["ch_roa"] = out["roa_t"] - g["roa_t"].shift(1)
    out["ch_fcf"] = (out["_fcf_raw"] - g["_fcf_raw"].shift(1)) / avg_assets
    out["tax"] = out["def_tax_exp"] / beg_assets
    out["leasedum"] = (out["op_lease_pv"] > 0).astype(int)
    out["oplease"] = (out["op_lease_pv"] - g["op_lease_pv"].shift(1)) / avg_assets
    out["pension"] = out["pension_ret"]
    out["ch_pension"] = out["pension_ret"] - g["pension_ret"].shift(1)

    out["_capex_lag1"] = g["capex"].shift(1)
    capex_3yr_avg = out.groupby("Stkcd")["_capex_lag1"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["exfin"] = (((out["cfo"] - capex_3yr_avg) / out["curr_assets"].replace(0, np.nan)) < -0.5).astype(int)
    out["cff"] = out["cff_net"] / avg_assets
    out["leverage"] = out["lt_debt"] / out["total_assets"]
    out["bm"] = out["equity"] / out["market_value"]
    out["ep"] = out["earnings"] / out["market_value"]

    drop_temp = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_temp)
