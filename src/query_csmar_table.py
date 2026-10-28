import os
import pandas as pd


def query_csmar_table(table_spec: dict, csmar_service, start_date: str, end_date: str, page_size: int = 200000) -> pd.DataFrame:
    """Query a single CSMAR table via csmarapi.CsmarService (with 200,000-row pagination) or load from data/raw/."""
    table_name = table_spec["table_name"]
    local_file = table_spec.get("local_file", f"data/raw/{table_name}.csv")
    columns = table_spec["columns"]
    base_cond = table_spec.get("condition", "1=1")

    # 1. If local cached raw CSV/XLSX already exists in data/raw/, load it directly to avoid CSMAR's 30-minute identical-query cooldown (py_api.txt Section 2.7)
    if os.path.exists(local_file) and os.path.getsize(local_file) > 100:
        print(f"  [Cache Hit] Loading existing raw table '{table_name}' from {local_file}...")
        if local_file.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(local_file)
        return pd.read_csv(local_file, low_memory=False)

    xlsx_alt = os.path.splitext(local_file)[0] + ".xlsx"
    if os.path.exists(xlsx_alt) and os.path.getsize(xlsx_alt) > 100:
        print(f"  [Cache Hit] Loading existing raw table '{table_name}' from {xlsx_alt}...")
        return pd.read_excel(xlsx_alt)

    # 2. Query live CSMAR Python API
    if csmar_service is not None:
        print(f"  [CSMAR API] Querying table '{table_name}' ({start_date} to {end_date})...")
        total_count = csmar_service.queryCount(columns, base_cond, table_name, start_date, end_date)
        try:
            total_rows = int(total_count)
        except (TypeError, ValueError):
            total_rows = page_size

        chunks = []
        offset = 0
        while offset < max(1, total_rows):
            paged_cond = f"{base_cond} limit {offset},{page_size}" if total_rows > page_size else base_cond
            df_chunk = csmar_service.query_df(columns, paged_cond, table_name, start_date, end_date)
            if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                chunks.append(df_chunk)
                print(f"    -> Fetched {len(df_chunk)} rows (offset={offset})")
                if len(df_chunk) < page_size:
                    break
            else:
                break
            offset += page_size

        if not chunks:
            raise RuntimeError(
                f"CSMAR API returned 0 rows (or no permission) for table '{table_name}' with condition '{base_cond}'. "
                f"Check csmar-log.log for details or verify that your CSMAR personal account has institutional access to '{table_name}'."
            )
        df_out = pd.concat(chunks, ignore_index=True)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        df_out.to_csv(local_file, index=False, encoding="utf-8")
        return df_out

    raise FileNotFoundError(
        f"Real CSMAR table '{table_name}' not found! Live csmarapi login was not ready (see data/api_verification_report.json) "
        f"and local file '{local_file}' does not exist."
    )
