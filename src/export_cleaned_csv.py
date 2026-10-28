import os
import pandas as pd


def export_cleaned_csv(df: pd.DataFrame, output_path: str) -> str:
    """Export the cleaned and winsorized feature matrix to a UTF-8 CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return os.path.abspath(output_path)
