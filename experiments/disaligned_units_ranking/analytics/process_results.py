import numpy as np
import pandas as pd

INPUT_CSV = "results_raw.csv"
OUTPUT_CSV = "results_processed.csv"

RANKINGS = ["replace_sweep_10_full", "interp_sweep_10_full"]
PROBE = "linear"

DEP_CONCEPT = "PassengerCar"
INDEP_CONCEPT = "ReinforcedCar"


def load(path):
    df = pd.read_csv(path)

    parts = df["out_dir"].str.split("/")
    df["ranking"] = parts.str[4]
    df["ranking_to_extract"] = parts.str[5].astype(int)

    df = df[df["ranking"].isin(RANKINGS) & (df["probe"] == PROBE)]
    return df[["seed", "out_concept", "protected_concept", "corr", "test_acc",
               "sensitivity_abs", "independence_abs", "ranking", "ranking_to_extract"]]


def aggregate(df):
    return (
        df.groupby(["corr", "ranking", "ranking_to_extract",
                    "out_concept", "protected_concept"], as_index=False)
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            sensitivity_abs_mean=("sensitivity_abs", "mean"),
            sensitivity_abs_std=("sensitivity_abs", "std"),
            independence_abs_mean=("independence_abs", "mean"),
            independence_abs_std=("independence_abs", "std"),
        )
    )


def compute_alignment(df):
    results = []

    for (corr, ranking, k), group in df.groupby(["corr", "ranking", "ranking_to_extract"]):
        dep = group[group["protected_concept"] == DEP_CONCEPT].iloc[0]
        indep = group[group["protected_concept"] == INDEP_CONCEPT].iloc[0]

        dep_mean, dep_std = dep["sensitivity_abs_mean"], dep["sensitivity_abs_std"]
        indep_mean, indep_std = indep["independence_abs_mean"], indep["independence_abs_std"]

        results.append({
            "corr": corr,
            "ranking": ranking,
            "ranking_to_extract": k,
            "alignment_mean": dep_mean * indep_mean,
            "alignment_std": np.sqrt(
                (indep_mean * dep_std) ** 2
                + (dep_mean * indep_std) ** 2
                + (dep_std * indep_std) ** 2
            ),
            "test_acc_mean": dep["test_acc_mean"],
            "test_acc_std": dep["test_acc_std"],
            "pc_mean": dep_mean,
            "pc_std": dep_std,
            "rc_mean": indep_mean,
            "rc_std": indep_std,
        })

    return pd.DataFrame(results)


def main():
    df = load(INPUT_CSV)
    df = aggregate(df)
    df = compute_alignment(df)

    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
