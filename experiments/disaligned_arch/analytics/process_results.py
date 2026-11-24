import numpy as np
import pandas as pd

from src.reliability.tnorms import tnorm_product

INPUT_CSV = "results_raw.csv"
OUTPUT_CSV = "results_processed.csv"

OUT_CONCEPT = "PassengerCar"
DEP_CONCEPT = "PassengerCar"
INDEP_CONCEPT = "ReinforcedCar"

probe_order = ["nn_linear", "nn_standard", "nn_medium", "nn_large"]


def load(path):
    df = pd.read_csv(path)
    df = df[df["out_concept"] == OUT_CONCEPT]
    df = df[["protected_concept", "corr", "probe", "test_acc",
             "sensitivity", "seed", "tr_loss"]].copy()
    df["sensitivity"] = df["sensitivity"].abs()
    df["tr_loss"] = df["tr_loss"].str.extract(r"tensor\(([^,]+)").astype(float)
    return df


def aggregate(df):
    df = (
        df.groupby(["protected_concept", "corr", "probe"], as_index=False)
        .agg(
            test_acc=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            sensitivity=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            tr_loss=("tr_loss", "mean"),
        )
    )
    df["independence"] = 1 - df["sensitivity"]
    df["independence_std"] = df["sensitivity_std"]
    return df


def add_alignment(df):
    results = []
    for _, g in df.groupby(["corr", "probe"]):
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

    return pd.concat(results)


def main():
    df = load(INPUT_CSV)
    df = aggregate(df)
    df = add_alignment(df)

    # order rows by probe and type of query
    df["probe"] = pd.Categorical(df["probe"], categories=probe_order, ordered=True)
    df["protected_concept"] = pd.Categorical(
        df["protected_concept"], categories=[DEP_CONCEPT, INDEP_CONCEPT], ordered=True
    )
    df = df.sort_values(["probe", "protected_concept"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()