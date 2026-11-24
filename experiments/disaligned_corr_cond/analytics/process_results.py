from ast import literal_eval

import pandas as pd

INPUT_CSV = "results_raw.csv"
OUTPUT_CSV = "results_processed.csv"

OUT_CONCEPT = "PassengerCar"


def parse_condition(s):
    c = literal_eval(s)
    if isinstance(c, list):
        val, concept = c[0][0], c[0][1]
        return f"{concept}={int(val)}"
    return "none"


def parse_corr(s):
    return literal_eval(s)[1]


def load(path):
    df = pd.read_csv(path)
    df["conditional"] = df["conditional"].fillna("None")
    df["condition"] = df["conditional"].map(parse_condition)
    df = df[df["protected_concept"] == OUT_CONCEPT]

    df["semantics"] = df["sensitivity"]
    return df


def curve(g, col):
    corr_grid = []
    means = []
    stds = []

    for corr, g in g.groupby("corr"):
        corr_grid.append(corr)
        means.append(g[col].mean())
        stds.append(g[col].std())
    return corr_grid, means, stds


def produce_data(df):
    data = []

    g = df[(df["protected_concept"] == OUT_CONCEPT) & (df["condition"] == "none")]
    corr_grid, means, stds = curve(g, "test_acc")
    data.append({
        "protected_concept": OUT_CONCEPT,
        "condition": "none",
        "metric": "test_acc",
        "corr_grid": corr_grid,
        "semantics_list": means,
        "semantics_std_list": stds,
    })

    for (protected_concept, condition), g in df.groupby(["protected_concept", "condition"]):
        corr_grid, means, stds = curve(g, "semantics")
        data.append({
            "protected_concept": protected_concept,
            "condition": condition,
            "metric": "semantics",
            "corr_grid": corr_grid,
            "semantics_list": means,
            "semantics_std_list": stds,
        })

    return data


def to_dataframe(data):
    rows = []
    for d in data:
        for corr, mean, std in zip(d["corr_grid"], d["semantics_list"], d["semantics_std_list"]):
            rows.append({
                "protected_concept": d["protected_concept"],
                "condition": d["condition"],
                "metric": d["metric"],
                "corr": float(corr),
                "mean": float(mean),
                "std": float(std),
            })
    return pd.DataFrame(rows)


def main():
    df = load(INPUT_CSV)
    data = produce_data(df)

    for d in data:
        print(d)

    out = to_dataframe(data)
    out.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    exit()
    main()