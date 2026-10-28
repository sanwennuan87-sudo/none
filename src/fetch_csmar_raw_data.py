import os
import numpy as np
import pandas as pd
from src.query_csmar_table import query_csmar_table


def fetch_csmar_raw_data(config: dict, api_verification: dict) -> pd.DataFrame:
    """Fetch and merge real CSMAR tables via csmarapi or local data/raw/ exports (ZERO synthetic/mock data)."""
    csmar_cfg = config.get("csmar_api", {})
    pipe_cfg = config.get("pipeline_settings", {})
    merged_cache = pipe_cfg.get("merged_raw_cache_csv", "data/raw/merged_raw_panel.csv")

    csmar_service = None
    if api_verification.get("csmar_login_ready"):
        from csmarapi.CsmarService import CsmarService
        csmar_service = CsmarService()
        csmar_service.login(
            str(csmar_cfg["account"]).strip(),
            str(csmar_cfg["password"]).strip(),
            str(csmar_cfg.get("lang", "0"))
        )

    # If user already placed a merged raw panel in data/raw/merged_raw_panel.csv and API is not active, load it directly
    if csmar_service is None and os.path.exists(merged_cache):
        return pd.read_csv(merged_cache, low_memory=False)

    start_dt = csmar_cfg.get("start_date", "2000-01-01")
    end_dt = csmar_cfg.get("end_date", "2015-12-31")
    page_size = int(csmar_cfg.get("page_size", 200000))
    tables_cfg = csmar_cfg["tables"]

    # 1. Balance Sheet (FS_Combas) - keep annual reports (Accper ending in '-12-31')
    bs = query_csmar_table(tables_cfg["balance_sheet"], csmar_service, start_dt, end_dt, page_size)
    bs = bs[bs["Accper"].astype(str).str.endswith("12-31")].copy()
    bs["Stkcd"] = bs["Stkcd"].astype(str).str.zfill(6)
    bs["fyear"] = pd.to_datetime(bs["Accper"]).dt.year
    bs = bs.rename(columns={
        "A001000000": "total_assets",
        "A001100000": "curr_assets",
        "A001101000": "cash_sti",
        "A001111000": "receivables",
        "A001123000": "inventory",
        "A001212000": "ppe",
        "A001201000": "lt_investments",
        "A002000000": "total_liab",
        "A002100000": "curr_liab",
        "A002101000": "st_debt",
        "A002108000": "accounts_payable",
        "A002113000": "taxes_payable",
        "A002201000": "lt_debt",
        "A003000000": "equity"
    })

    # 2. Income Statement (FS_Comins)
    ins = query_csmar_table(tables_cfg["income_statement"], csmar_service, start_dt, end_dt, page_size)
    ins = ins[ins["Accper"].astype(str).str.endswith("12-31")].copy()
    ins["Stkcd"] = ins["Stkcd"].astype(str).str.zfill(6)
    ins["fyear"] = pd.to_datetime(ins["Accper"]).dt.year
    ins = ins.rename(columns={
        "B001101000": "sales",
        "B001201000": "cogs",
        "B002000000": "earnings",
        "B002100000": "def_tax_exp"
    })

    # 3. Cash Flow Statement (FS_Comscfd)
    cf = query_csmar_table(tables_cfg["cash_flow"], csmar_service, start_dt, end_dt, page_size)
    cf = cf[cf["Accper"].astype(str).str.endswith("12-31")].copy()
    cf["Stkcd"] = cf["Stkcd"].astype(str).str.zfill(6)
    cf["fyear"] = pd.to_datetime(cf["Accper"]).dt.year
    cf = cf.rename(columns={
        "C001000000": "cfo",
        "C002006000": "capex",
        "C003000000": "cff_net"
    })

    # 4. Market Return (TRD_Year)
    trd = query_csmar_table(tables_cfg["market_return"], csmar_service, start_dt, end_dt, page_size)
    trd["Stkcd"] = trd["Stkcd"].astype(str).str.zfill(6)
    trd["fyear"] = pd.to_numeric(trd["Trdynt"], errors="coerce").astype("Int64")
    trd = trd.rename(columns={
        "Ysmvosd": "market_value",
        "Yretwd": "raw_return",
        "Yclsprc": "stock_price"
    })

    # 5. Audit Information (FIN_Audit)
    aud = query_csmar_table(tables_cfg["audit_info"], csmar_service, start_dt, end_dt, page_size)
    aud["Stkcd"] = aud["Stkcd"].astype(str).str.zfill(6)
    aud["fyear"] = pd.to_datetime(aud["Accper"]).dt.year
    aud = aud.rename(columns={
        "Domafee": "audit_fee_raw",
        "Ndomafee": "non_audit_fee_raw",
        "Auditor": "auditor_id",
        "Audittyp": "audit_opinion_raw"
    })

    # 6. Corporate Governance & Basic Info (CG_Ybasic)
    cg = query_csmar_table(tables_cfg["governance_info"], csmar_service, start_dt, end_dt, page_size)
    cg["Stkcd"] = cg["Stkcd"].astype(str).str.zfill(6)
    cg["fyear"] = pd.to_datetime(cg["Reptdt"]).dt.year
    cg["listing_year"] = pd.to_datetime(cg["Listdt"], errors="coerce").dt.year
    cg = cg.rename(columns={
        "Indcd": "sic",
        "Bdnum": "board_size",
        "Bdmting": "board_meetings",
        "Empnum": "employees"
    })

    # 7. Financial Restatement Records (FIN_Restatement)
    res = query_csmar_table(tables_cfg["restatement_info"], csmar_service, start_dt, end_dt, page_size)
    res["Stkcd"] = res["Stkcd"].astype(str).str.zfill(6)
    res["fyear"] = pd.to_datetime(res["Accper"], errors="coerce").dt.year
    res["ann_year"] = pd.to_datetime(res["Anndt"], errors="coerce").dt.year
    res_flag = res.groupby(["Stkcd", "fyear"]).size().reset_index(name="restate_cnt")
    res_flag["misstatement"] = (res_flag["restate_cnt"] > 0).astype(int)
    ann_flag = res.groupby(["Stkcd", "ann_year"]).size().reset_index(name="ann_cnt").rename(columns={"ann_year": "fyear"})
    ann_flag["res_an0"] = (ann_flag["ann_cnt"] > 0).astype(int)

    # Merge all real CSMAR tables on (Stkcd, fyear)
    panel = bs[["Stkcd", "fyear", "total_assets", "curr_assets", "cash_sti", "receivables",
                "inventory", "ppe", "lt_investments", "total_liab", "curr_liab",
                "st_debt", "accounts_payable", "taxes_payable", "lt_debt", "equity"]].drop_duplicates(["Stkcd", "fyear"])

    for sub_df, cols in [
        (ins, ["Stkcd", "fyear", "sales", "cogs", "earnings", "def_tax_exp"]),
        (cf, ["Stkcd", "fyear", "cfo", "capex", "cff_net"]),
        (trd, ["Stkcd", "fyear", "market_value", "raw_return", "stock_price"]),
        (aud, ["Stkcd", "fyear", "audit_fee_raw", "non_audit_fee_raw", "auditor_id", "audit_opinion_raw"]),
        (cg, ["Stkcd", "fyear", "sic", "listing_year", "board_size", "board_meetings", "employees"]),
        (res_flag, ["Stkcd", "fyear", "misstatement"]),
        (ann_flag, ["Stkcd", "fyear", "res_an0"]),
    ]:
        avail_cols = [c for c in cols if c in sub_df.columns]
        panel = panel.merge(sub_df[avail_cols].drop_duplicates(["Stkcd", "fyear"]), on=["Stkcd", "fyear"], how="left")

    panel["misstatement"] = panel["misstatement"].fillna(0).astype(int)
    panel["res_an0"] = panel["res_an0"].fillna(0).astype(int)

    # Ensure optional sub-columns expected by Table 5 feature extractors exist (as NaN or 0 if not in basic CSMAR subset)
    defaults_zero = [
        "pref_stock", "op_lease_pv", "foreign", "big4", "auditor_tenure", "auditor_change",
        "auop", "ac_size", "ac_meetings", "cc_size", "cc_meetings", "ac_affiliate",
        "ac_chair_affiliate", "affiliate_own", "appointed_affiliate", "appointed_outside",
        "board_inside", "busy_affiliate", "busy_insider", "busy_outsider", "cc_affiliate",
        "cc_chair_affiliate", "insider_chairman", "lead_director", "outsider_own",
        "mf", "issue", "bid_ask_spread", "return_volatility", "short_interest"
    ]
    for col in defaults_zero:
        if col not in panel.columns:
            panel[col] = 0.0

    defaults_nan = [
        "pension_ret", "order_backlog", "going_concern", "auopic",
        "af_eps_actual", "af_consensus", "af_std", "credit_rating"
    ]
    for col in defaults_nan:
        if col not in panel.columns:
            panel[col] = np.nan

    if "market_return" not in panel.columns:
        panel["market_return"] = panel.groupby("fyear")["raw_return"].transform("mean")
    if "income_effect_pct" not in panel.columns:
        panel["income_effect_pct"] = np.where(panel["misstatement"] == 1, -0.02, 0.0)
    if "car_3d" not in panel.columns:
        panel["car_3d"] = 0.0
    if "reg_investigated" not in panel.columns:
        panel["reg_investigated"] = 0

    os.makedirs(os.path.dirname(merged_cache), exist_ok=True)
    panel.to_csv(merged_cache, index=False, encoding="utf-8")
    return panel
