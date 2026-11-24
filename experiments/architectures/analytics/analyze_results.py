import numpy as np
import pandas as pd

from src.reliability.tnorms import tnorm_min

INPUT_CSV = "results.csv"
OUTPUT_CSV = "aggregated_df.csv"

CONCEPTS = [
    "FreightWagon", "WarTrain", "EmptyTrain", "ReinforcedCar", "PassengerTrain",
    "LongTrain", "FreightTrain", "LongWagon", "RuralTrain", "MixedTrain", "OpenRoofCar",
]

PROBES = ["linear", "ridge", "lgbm", "nn_standard", "nn_medium"]

CONCEPT_INDEP = {
    "FreightWagon": {
        "dep": ["FreightWagon"],
        "indep": ["PassengerCar", "ReinforcedCar", "EmptyWagon", "PassengerLongWagon", "notLongWagon"],
    },
    "WarTrain": {
        "dep": ["WarTrain", "PassengerCar", "ReinforcedCar"],
        "indep": ["EmptyWagon", "notLongWagon", "FreightWagon"],
    },
    "EmptyTrain": {
        "dep": ["EmptyTrain", "EmptyWagon"],
        "indep": ["notLongWagon"],
    },
    "ReinforcedCar": {
        "dep": ["ReinforcedCar"],
        "indep": ["PassengerCar", "EmptyWagon", "PassengerLongWagon", "notLongWagon", "FreightWagon", "notEmptyTrain"],
    },
    "PassengerTrain": {
        "dep": ["PassengerTrain"],
        "indep": ["ReinforcedCar", "EmptyWagon", "notLongWagon", "FreightWagon"],
    },
    "LongTrain": {
        "dep": ["LongTrain"],
        "indep": ["PassengerCar", "ReinforcedCar", "EmptyWagon", "FreightWagon", "notEmptyTrain"],
    },
    "FreightTrain": {
        "dep": ["FreightTrain"],
        "indep": ["PassengerCar", "ReinforcedCar", "EmptyWagon", "PassengerLongWagon", "notLongWagon"],
    },
    "LongWagon": {
        "dep": ["LongWagon"],
        "indep": ["PassengerCar", "ReinforcedCar", "EmptyWagon", "FreightWagon", "notEmptyTrain"],
    },
    "RuralTrain": {
        "dep": ["RuralTrain"],
        "indep": ["ReinforcedCar"],
    },
    "MixedTrain": {
        "dep": ["MixedTrain"],
        "indep": ["ReinforcedCar", "notLongWagon"],
    },
    "OpenRoofCar": {
        "dep": ["OpenRoofCar"],
        "indep": ["PassengerCar", "ReinforcedCar", "EmptyWagon", "PassengerLongWagon", "notLongWagon", "notEmptyTrain", "FreightWagon"],
    },
}


def compute_alignment(df, concept):
    values = []
    for dep in CONCEPT_INDEP[concept]["dep"]:
        value = df.loc[df["protected_concept"] == dep, "sensitivity"].dropna().iloc[0]
        values.append(value)

    for indep in CONCEPT_INDEP[concept]["indep"]:
        value = df.loc[df["protected_concept"] == indep, "independence"].dropna().iloc[0]
        values.append(value)

    return tnorm_min(values)


def summarize(df):
    rows = []
    for train_type in sorted(df["model_cfg"].unique()):
        df_model = df[df["model_cfg"] == train_type]

        for concept in CONCEPTS:
            df_concept = df_model[df_model["out_concept"] == concept]

            for probe in PROBES:
                df_probe = df_concept[df_concept["probe"] == probe]

                acc_values = []
                alignment_values = []
                for seed in sorted(df_probe["seed"].unique()):
                    df_seed = df_probe[df_probe["seed"] == seed]
                    acc_values.append(df_seed["test_acc"].iloc[0])
                    alignment_values.append(compute_alignment(df_seed, concept))

                rows.append({
                    "train_type": train_type,
                    "out_concept": concept,
                    "probe": probe,
                    "acc_mean": float(np.mean(acc_values)),
                    "acc_std": float(np.std(acc_values)),
                    "alignment_mean": float(np.mean(alignment_values)),
                    "alignment_std": float(np.std(alignment_values)),
                })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_CSV)
    df["sensitivity"] = df["sensitivity"].abs()
    df["independence"] = 1 - df["sensitivity"]

    summary = summarize(df)
    summary.to_csv(OUTPUT_CSV)


if __name__ == "__main__":
    main()