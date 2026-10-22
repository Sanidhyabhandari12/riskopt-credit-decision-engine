"""
Step 1 + 2: load the raw CSVs, audit data quality, and build the
model table with SQL (DuckDB).

Why DuckDB? It runs SQL directly on CSV files with no database server,
and handles the 1.7M-row bureau table in a couple of seconds.
"""
import duckdb
import pandas as pd

from config import SQL_DIR, PROCESSED_DIR, OUTPUT_DIR, ID_COL, TARGET_COL, CATEGORICAL_COLS

RAW_TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
}

FEATURE_SQL_FILES = [
    "01_bureau_features.sql",
    "02_previous_application_features.sql",
    "03_build_model_table.sql",
]


def load_raw_tables(con, data_dir):
    """Register each CSV as a DuckDB table."""
    for table, filename in RAW_TABLES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download it from Kaggle (Home Credit Default Risk) "
                f"or run scripts/make_sample_data.py and use --sample."
            )
        # sample_size=-1: scan the whole file to detect column types, so a column that
        # is empty in the first few thousand rows is not wrongly read as text
        con.execute(f"CREATE OR REPLACE TABLE {table} AS "
                    f"SELECT * FROM read_csv_auto('{path}', sample_size=-1)")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  loaded {table:<22} {n:>10,} rows")


def audit_application_table(con):
    """Basic data-quality checks on the main table. Saves a missing-value report."""
    app = con.execute("SELECT * FROM application").df()
    audit_dir = OUTPUT_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    missing = (app.isna().mean() * 100).round(2).sort_values(ascending=False)
    missing = missing[missing > 0].rename("missing_pct").to_frame()
    missing.to_csv(audit_dir / "missing_values.csv")

    dup_ids = app[ID_COL].duplicated().sum()
    bad_rate = app[TARGET_COL].mean()
    employed_anomaly = (app["DAYS_EMPLOYED"] == 365243).mean()

    summary = {
        "rows": len(app),
        "columns": app.shape[1],
        "duplicate_ids": int(dup_ids),
        "bad_rate": round(float(bad_rate), 4),
        "columns_with_missing": int(len(missing)),
        "days_employed_365243_share": round(float(employed_anomaly), 4),
    }
    pd.Series(summary).to_csv(audit_dir / "audit_summary.csv", header=["value"])

    print(f"  rows={summary['rows']:,}  cols={summary['columns']}  duplicate ids={dup_ids}")
    print(f"  bad rate={bad_rate:.2%}  (imbalanced - roughly 1 in {round(1 / bad_rate)} loans)")
    print(f"  {len(missing)} columns have missing values; top 5:")
    for col, pct in missing["missing_pct"].head(5).items():
        print(f"     {col:<28} {pct:.1f}%")
    print(f"  DAYS_EMPLOYED = 365243 placeholder in {employed_anomaly:.1%} of rows -> set to NULL + flag")

    assert dup_ids == 0, "SK_ID_CURR should be unique in application_train"
    return summary


def build_model_table(con):
    """Run the feature SQL files in order and return the final model table."""
    for sql_file in FEATURE_SQL_FILES:
        con.execute((SQL_DIR / sql_file).read_text())
        print(f"  ran {sql_file}")

    # join validation: LEFT JOINs on aggregated tables must not duplicate applicants
    n_app = con.execute("SELECT COUNT(*) FROM application").fetchone()[0]
    n_model = con.execute("SELECT COUNT(*) FROM model_table").fetchone()[0]
    n_unique = con.execute(f"SELECT COUNT(DISTINCT {ID_COL}) FROM model_table").fetchone()[0]
    assert n_app == n_model == n_unique, f"join changed row count: {n_app} -> {n_model}"
    print(f"  join check passed: {n_model:,} rows, one per applicant")

    df = con.execute("SELECT * FROM model_table").df()

    # treat missing categories as their own level ("Missing" is often informative)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Missing").astype(str)

    return df


def run(data_dir):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()   # in-memory database

    print("\n[1] Loading raw data")
    load_raw_tables(con, data_dir)

    print("\n[2] Data quality audit")
    audit_application_table(con)

    print("\n[3] SQL feature engineering")
    df = build_model_table(con)
    df.to_parquet(PROCESSED_DIR / "model_table.parquet", index=False)
    print(f"  model table: {df.shape[0]:,} rows x {df.shape[1]} columns "
          f"-> data/processed/model_table.parquet")

    return con, df
