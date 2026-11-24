import numpy as np
import pandas as pd

INPUT_CSV = "results_raw.csv"
OUTPUT_CSV = "results_processed.csv"

concept_corr_pair_dict = {
    "PassengerCar1": ("PassengerCar", "ReinforcedCar"),
    "PassengerCar": ("PassengerCar", "OpenRoofCar"),
    "WarTrain": ("PassengerCar", "FreightWagon"),
    "OpenRoofCar": ("OpenRoofCar", "PassengerCar"),
}

COLUMNS = ["out_concept", "protected_concept", "corr_pair", "corr", "tr_acc",
           "val_acc", "test_acc", "sensitivity", "sensitivity_abs", "cf_acc",
           "intervention_mode", "seed"]


def curve(col, corr_levels, df, concept=None):
    means = []
    stds = []
    for c in corr_levels:
        _df = df[df["corr"] == c]
        if concept is not None:
            _df = _df[_df["protected_concept"] == concept]
        vals = _df[col].dropna()
        means.append(vals.mean())
        stds.append(vals.std())
    return np.array(means), np.array(stds)


def process_group(df_corr, concept):
    concept_dep, concept_indep = concept_corr_pair_dict[concept]
    corr_levels = np.sort(df_corr["corr"].unique())

    acc, acc_std = curve("test_acc", corr_levels, df_corr)
    sens, sens_std = curve("sensitivity", corr_levels, df_corr, concept_dep)
    indep, indep_std = curve("independence", corr_levels, df_corr, concept_indep)

    alg = sens * indep
    alg_std = np.sqrt((indep * sens_std) ** 2 +
                      (sens * indep_std) ** 2 +
                      (sens_std * indep_std) ** 2)

    return pd.DataFrame({
        "corr_pair": df_corr["corr_pair"].iloc[0],
        "out_concept": concept,
        "concept_dep": concept_dep,
        "concept_indep": concept_indep,
        "corr": corr_levels,
        "acc_mean": acc, "acc_std": acc_std,
        "sens_mean": sens, "sens_std": sens_std,
        "indep_mean": indep, "indep_std": indep_std,
        "alg_mean": alg, "alg_std": alg_std,
    })


def main():
    df = pd.read_csv(INPUT_CSV)[COLUMNS]
    df["sensitivity"] = df["sensitivity"].abs()
    df["independence"] = 1 - df["sensitivity"]

    frames = []
    for pair, df_corr in df.groupby("corr_pair"):
        concept = df_corr["out_concept"].unique()[0]
        frames.append(process_group(df_corr, concept))

    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()