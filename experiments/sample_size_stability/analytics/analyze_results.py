import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.reliability.tnorms import tnorm_min

INPUT_CSV = "results_full_grid.csv"

CONCEPTS = [
    "FreightWagon", "WarTrain", "EmptyTrain", "ReinforcedCar", "PassengerTrain",
    "LongTrain", "FreightTrain", "LongWagon", "RuralTrain", "MixedTrain", "OpenRoofCar",
]


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

CONCEPT_NAMES = {
    "FreightWagon": r"$\mathsf{\exists has.FreightWagon}$",
    "WarTrain": r"$\mathsf{WarTrain}$",
    "EmptyTrain": r"$\mathsf{EmptyTrain}$",
    "ReinforcedCar": r"$\mathsf{\exists has.ReinforcedCar}$",
    "PassengerTrain": r"$\mathsf{PassengerTrain}$",
    "LongTrain": r"$\mathsf{LongTrain}$",
    "FreightTrain": r"$\mathsf{FreightTrain}$",
    "LongWagon": r"$\mathsf{\exists has.LongWagon}$",
    "RuralTrain": r"$\mathsf{RuralTrain}$",
    "MixedTrain": r"$\mathsf{MixedTrain}$",
    "OpenRoofCar": r"$\mathsf{\exists has.OpenRoofCar}$",
}

MODEL_NAMES = {
    "type_a": "$f_A$",
    "type_b": "$f_B$",
    "type_c": "$f_C$",
}


GRID = list(range(1, 501))
PLOT_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500]

def styling():
    sns.set_theme(context="paper", style="ticks", palette="deep",
                  font_scale=1, color_codes=True)
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\newcommand{\indep}{\mathrel{\perp\!\!\!\perp}}"
    )
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.width": 0.2,
        "xtick.major.size": 3,
        "ytick.major.width": 0.2,
        "ytick.major.size": 3,
        "lines.linewidth": 0.7,
    })


def compute_alignment(df, concept):
    df = df[df["out_concept"] == concept]

    query_curves = []
    for dep in CONCEPT_INDEP[concept]["dep"]:
        text = df.loc[df["protected_concept"] == dep, "sensitivity"].dropna().iloc[0]
        values = ast.literal_eval(text)
        query_curves.append([abs(value) for value in values])

    for indep in CONCEPT_INDEP[concept]["indep"]:
        text = df.loc[df["protected_concept"] == indep, "independence"].dropna().iloc[0]
        query_curves.append(ast.literal_eval(text))

    alignment = []
    for i in range(len(GRID)):
        values_at_n = [query_curve[i] for query_curve in query_curves]
        alignment.append(tnorm_min(values_at_n))
    return alignment


def collect_curves(df):
    curves = {}
    for model, df_model in df.groupby("model_cfg"):
        curves[model] = {}
        for concept in CONCEPTS:
            seed_curves = []
            for seed, df_seed in df_model.groupby("seed"):
                seed_curves.append(compute_alignment(df_seed, concept))
            curves[model][concept] = np.array(seed_curves)
    return curves


def overall_curve(curves, stat):
    assert stat in ("mean", "std")

    pair_curves = []
    for model in curves:
        for concept in CONCEPTS:
            seed_curves = curves[model][concept]
            if stat == "mean":
                pair_curves.append(seed_curves.mean(axis=0))
            else:
                pair_curves.append(seed_curves.std(axis=0, ddof=1))

    return np.mean(pair_curves, axis=0)


def plot_curves(curves, ncols=4):
    nrows = int(np.ceil(len(CONCEPTS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 1.5 * nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, concept in enumerate(CONCEPTS):
        ax = axes[i]
        row = i // ncols
        col = i % ncols

        for model in curves:
            mean = curves[model][concept].mean(axis=0)
            ax.plot(GRID, mean, label=MODEL_NAMES[model])

        ax.set_xlim(1, 500)
        ax.set_ylim(0, 1)
        ax.set_title(CONCEPT_NAMES[concept], fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", left=True, labelleft=True, labelsize=10)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        if row == nrows - 1 or (row == 1 and col == 3):
            ax.set_xticks([1, 100, 200, 300, 400, 500])
            ax.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=10)
        else:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

    for ax in axes[len(CONCEPTS):]:
        ax.set_visible(False)

    fig.text(0.02, 0.52, "Alignment", fontsize=13, va="center", ha="center", rotation="vertical")
    fig.text(0.54, 0.03, "Test Samples", fontsize=13, va="center", ha="center")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.08),
               frameon=False, fontsize=15)

    plt.subplots_adjust(top=0.945, bottom=0.105, left=0.075, right=0.980, hspace=0.365, wspace=0.260)
    plt.show()


def plot_overall(overall, grid, stat):
    values = [overall[n - 1] for n in grid]

    fig, ax = plt.subplots(figsize=(4, 2.3))
    ax.plot(grid, values)

    ax.set_xlim(-9, 502)
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_xlabel("Test Samples")
    if stat == "std":
        ax.set_ylabel("Standard Deviation")
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    else:
        ax.set_ylabel("Mean")
        ax.set_ylim(0.2, 0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(top=0.970, bottom=0.192, left=0.144, right=0.970)
    plt.show()


def main():
    styling()
    df = pd.read_csv(INPUT_CSV)

    curves = collect_curves(df)
    overall_std = overall_curve(curves, "std")
    overall_mean = overall_curve(curves, "mean")

    plot_curves(curves)
    plot_overall(overall_std, PLOT_GRID, "std")
    plot_overall(overall_mean, GRID, "mean")

    std_1 = overall_std[1 - 1]
    std_100 = overall_std[100 - 1]
    std_500 = overall_std[500 - 1]

    total_reduction = std_1 - std_500
    reduction_at_100 = std_1 - std_100

    print(f"Overall std at N=1:   {std_1:.4f}")
    print(f"Overall std at N=100: {std_100:.4f}")
    print(f"Overall std at N=500: {std_500:.4f}")
    print(f"Total reduction: {total_reduction:.4f}")
    print(f"% of total at N=100:   {reduction_at_100 / total_reduction:.1%}")


if __name__ == "__main__":
    main()