import numpy as np
import pandas as pd

from src.reliability.tnorms import tnorm_product

INPUT_CSV = "results_raw.csv"
OUTPUT_CSV = "results_processed.csv"

DEP_CONCEPT = "PassengerCar"
INDEP_CONCEPT = "ReinforcedCar"

TO_EXTRACT_ORDER = [
    "{'36': [4]}",
    "{'36': [12]}",
    "{'36': [4, 12]}",
]


def add_alignment(df):
    results = []
    for _, g in df.groupby(["probe", "corr", "to_extract"]):
        dep_row = g[g["protected_concept"] == DEP_CONCEPT].iloc[0]
        indep_row = g[g["protected_concept"] == INDEP_CONCEPT].iloc[0]

        dep, dep_std = dep_row["sensitivity"], dep_row["sensitivity_std"]
        indep, indep_std = indep_row["independence"], indep_row["independence_std"]

        g = g.copy()
        g["alg"] = tnorm_product([dep, indep])
        g["alg_std"] = np.sqrt(
            dep**2 * indep_std**2
            + indep**2 * dep_std**2
            + indep_std**2 * dep_std**2
        )
        results.append(g)

    return pd.concat(results).sort_index()


def main():
    df = pd.read_csv(INPUT_CSV)

    df = (
        df.groupby(["probe", "protected_concept", "corr", "to_extract"], as_index=False)
        .agg(
            test_acc=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            sensitivity=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
        )
    )
    df["independence"] = 1 - df["sensitivity"]
    df["independence_std"] = df["sensitivity_std"]

    df = add_alignment(df)

    df = df.sort_values(
        "to_extract",
        key=lambda col: col.map(TO_EXTRACT_ORDER.index),
        kind="stable",
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()