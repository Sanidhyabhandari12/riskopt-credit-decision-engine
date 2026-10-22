"""
Step 3: exploratory analytics.

Runs the business queries in sql/04_portfolio_analysis.sql and saves
each result as a CSV and a simple bar chart of the bad rate.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SQL_DIR, OUTPUT_DIR


def read_named_queries(path):
    """Split a .sql file into {name: query} using '-- name: xxx' markers."""
    queries = {}
    current_name = None
    for line in path.read_text().splitlines():
        if line.strip().startswith("-- name:"):
            current_name = line.split(":", 1)[1].strip()
            queries[current_name] = []
        elif current_name:
            queries[current_name].append(line)
    return {name: "\n".join(lines).strip().rstrip(";") for name, lines in queries.items()}


def plot_bad_rate(df, name, out_dir):
    label_col = df.columns[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[label_col].astype(str), df["bad_rate_pct"], color="#2a78d6")
    ax.set_title(name.replace("_", " ").title())
    ax.set_ylabel("Bad rate (%)")
    ax.set_xlabel(label_col.replace("_", " "))
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def run(con):
    print("\n[4] Exploratory analytics (SQL)")
    out_dir = OUTPUT_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, query in read_named_queries(SQL_DIR / "04_portfolio_analysis.sql").items():
        df = con.execute(query).df()
        df.to_csv(out_dir / f"{name}.csv", index=False)
        if "bad_rate_pct" in df.columns and len(df) > 1:
            plot_bad_rate(df, name, out_dir)
        results[name] = df
        print(f"  {name:<36} {len(df)} rows")

    s = results["portfolio_summary"].iloc[0]
    print(f"  portfolio: {int(s['applications']):,} applications, bad rate {s['bad_rate_pct']}%")
    return results
