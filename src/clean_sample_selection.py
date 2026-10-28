import pandas as pd


def clean_sample_selection(raw_df: pd.DataFrame, start_year: int = 2001, end_year: int = 2014) -> pd.DataFrame:
    """Apply Table 1 sample selection criteria from Bertomeu et al. (2020) including fiscal years 2001-2014."""
    df = raw_df.sort_values(["Stkcd", "fyear"]).reset_index(drop=True).copy()

    # Require positive total assets in the prior year (two consecutive years of nonmissing positive assets)
    df["lag_total_assets"] = df.groupby("Stkcd")["total_assets"].shift(1)
    df["lag_raw_return"] = df.groupby("Stkcd")["raw_return"].shift(1)

    # Filter firm-years between fiscal year 2001-2014 (Table 1 of Bertomeu et al. 2020)
    year_mask = df["fyear"].between(start_year, end_year)

    valid_mask = (
        year_mask
        & df["total_assets"].notna()
        & (df["total_assets"] > 0)
        & df["lag_total_assets"].notna()
        & (df["lag_total_assets"] > 0)
        & df["lag_raw_return"].notna()
    )
    df = df.loc[valid_mask].copy()

    # Keep material misstatements (|income_effect| > 1%, 3-day CAR < -10%, or regulatory investigation)
    material_cond = (
        (df["misstatement"] == 0)
        | (df["income_effect_pct"].abs() > 0.01)
        | (df["car_3d"] < -0.10)
        | (df["reg_investigated"] == 1)
    )
    df = df.loc[material_cond].copy()

    # Remove firm-years that include both an ongoing misstatement and a restatement announcement in the same year
    same_year_announce = (df["misstatement"] == 1) & (df["res_an0"] == 1)
    df = df.loc[~same_year_announce].reset_index(drop=True)

    return df
