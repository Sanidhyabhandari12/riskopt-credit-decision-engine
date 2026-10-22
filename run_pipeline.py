"""
RiskOpt - end-to-end pipeline.

    python run_pipeline.py            # real Kaggle data in data/raw/
    python run_pipeline.py --sample   # synthetic sample in data/sample/

Steps:
    data load + audit -> SQL features -> SQL EDA -> train models ->
    calibrate -> explain -> profit model -> threshold optimisation -> back-test
"""
import argparse
import time

from config import OUTPUT_DIR, RAW_DIR, SAMPLE_DIR
from src import data_pipeline, decision_engine, eda, modeling, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="use the synthetic sample data")
    args = parser.parse_args()

    start = time.time()
    data_dir = SAMPLE_DIR if args.sample else RAW_DIR
    print(f"RiskOpt pipeline - data from {data_dir}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con, model_table = data_pipeline.run(data_dir)
    eda.run(con)
    champion, scored_calib, scored_test, metrics = modeling.run(model_table)
    decision, strategies, scored_test = decision_engine.run(scored_calib, scored_test)

    # files the dashboard reads
    scored_calib.to_parquet(OUTPUT_DIR / "scored_calib.parquet", index=False)
    scored_test.to_parquet(OUTPUT_DIR / "scored_test.parquet", index=False)
    (OUTPUT_DIR / "data_source.txt").write_text("sample" if args.sample else "kaggle")

    report.run(metrics, decision, strategies, is_sample=args.sample)
    print(f"\nDone in {time.time() - start:.0f}s. Launch the dashboard with:  "
          f"streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
