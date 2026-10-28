import numpy as np
import pandas as pd


def extract_market_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market, analyst, credit rating, business, and 15 SIC industry dummy variables from Table 5."""
    out = df.sort_values(["Stkcd", "fyear"]).copy()
    g = out.groupby("Stkcd")

    # Market-adjusted returns
    out["ret_t"] = out["raw_return"] - out["market_return"]
    out["ret_t_1"] = g["ret_t"].shift(1)

    # Analyst variables
    out["beat"] = (
        out["af_consensus"].notna() & (out["af_eps_actual"] >= out["af_consensus"])
    ).astype(int)
    out["af_error"] = (out["af_eps_actual"] - out["af_consensus"]).abs() / out["stock_price"].replace(0, np.nan)
    out["disper"] = out["af_std"] / out["af_consensus"].abs().replace(0, np.nan)

    # Liquidity, volatility, and short interest
    out["spread"] = out["bid_ask_spread"]
    out["vol"] = out["return_volatility"]
    out["short"] = out["short_interest"]

    # Credit rating variables (Avramov et al. 2009; Section 2 p. 476)
    out["rating_chg"] = out["credit_rating"] - g["credit_rating"].shift(1)
    out["rating_downgrade"] = (out["rating_chg"] > 0).astype(int)

    # Business environment variables
    out["age"] = out["fyear"] - out["listing_year"]
    pct_ch_emp = (out["employees"] - g["employees"].shift(1)) / g["employees"].shift(1).replace(0, np.nan)
    pct_ch_assets = (out["total_assets"] - g["total_assets"].shift(1)) / g["total_assets"].shift(1).replace(0, np.nan)
    out["ch_emp"] = pct_ch_emp - pct_ch_assets

    pct_ch_backlog = (out["order_backlog"] - g["order_backlog"].shift(1)) / g["order_backlog"].shift(1).replace(0, np.nan)
    pct_ch_sales = (out["sales"] - g["sales"].shift(1)) / g["sales"].shift(1).replace(0, np.nan)
    out["ch_backlog"] = pct_ch_backlog - pct_ch_sales

    # Past restatement announcements
    out["res_an1"] = g["res_an0"].shift(1).fillna(0).astype(int)
    out["res_an2"] = g["res_an0"].shift(2).fillna(0).astype(int)
    out["_res_lag3"] = g["res_an0"].shift(3)
    out["res_an3"] = (
        out.groupby("Stkcd")["_res_lag3"]
        .rolling(5, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(int)
    )
    out = out.drop(columns=["_res_lag3"])

    # 15 SIC major industry indicators (Table 5)
    sic = out["sic"]
    out["ind1"] = sic.between(100, 999).astype(int)
    out["ind2"] = (sic.between(1000, 1299) | sic.between(1400, 1999)).astype(int)
    out["ind3"] = sic.between(2000, 2141).astype(int)
    out["ind4"] = sic.between(2200, 2399).astype(int)
    out["ind5"] = sic.between(2400, 2796).astype(int)
    out["ind6"] = (sic.between(2800, 2824) | sic.between(2840, 2899)).astype(int)
    out["ind7"] = (sic.between(1300, 1399) | sic.between(2900, 2999)).astype(int)
    out["ind8"] = (
        sic.between(3000, 3569) | sic.between(3580, 3669) | sic.between(3680, 3999)
    ).astype(int)
    out["ind9"] = (
        sic.between(3570, 3579) | sic.between(3670, 3679) | sic.between(7370, 7379)
    ).astype(int)
    out["ind10"] = sic.between(4000, 4899).astype(int)
    out["ind11"] = sic.between(4900, 4999).astype(int)
    out["ind12"] = sic.between(5000, 5999).astype(int)
    out["ind13"] = (sic.between(7000, 7369) | sic.between(7380, 9999)).astype(int)
    out["ind14"] = sic.between(6000, 6999).astype(int)
    out["ind15"] = (sic.between(2830, 2836) | sic.between(3829, 3851)).astype(int)

    return out
