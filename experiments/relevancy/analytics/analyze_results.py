import pprint
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

from src.reliability.tnorms import tnorm_min

INPUT_CSV = "results.csv"

CONCEPTS = [
    "FreightWagon", "WarTrain", "EmptyTrain", "ReinforcedCar", "PassengerTrain",
    "LongTrain", "FreightTrain", "LongWagon", "RuralTrain", "MixedTrain", "OpenRoofCar",
]

LETTERS = list(string.ascii_lowercase[:len(CONCEPTS)])

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

RELEVANCY = {
    "type_a": ["FreightWagon", "WarTrain", "EmptyTrain", "ReinforcedCar"],
    "type_b": ["FreightWagon", "PassengerTrain", "LongTrain", "FreightTrain", "LongWagon"],
    "type_c": ["FreightWagon", "LongWagon", "RuralTrain", "MixedTrain"],
}

MODEL_NAMES = {
    "type_a": "$f_A$",
    "type_b": "$f_B$",
    "type_c": "$f_C$",
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

ACC_COLOR = sns.color_palette("deep")[0]
ALIGN_COLOR = sns.color_palette("deep")[1]
BAR_WIDTH = 0.45
GROUP_SPACING = 1.1
ERROR_KW = dict(lw=0.5, capsize=3, capthick=0.5)


def styling():
    sns.set_theme(context="paper", style="ticks", palette="deep",
                  font_scale=1, color_codes=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
        "axes.linewidth": 0.5,
        "axes.labelsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.major.width": 0.2,
        "xtick.major.size": 3,
        "ytick.major.width": 0.2,
        "ytick.major.size": 3,
        "lines.linewidth": 0.8,
    })


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
    data = {}
    for model in sorted(df["model_cfg"].unique()):
        df_model = df[df["model_cfg"] == model]

        acc_mean = []
        acc_std = []
        alignment_mean = []
        alignment_std = []

        for concept in CONCEPTS:
            df_concept = df_model[df_model["out_concept"] == concept]

            acc_values = []
            alignment_values = []
            for seed in sorted(df_concept["seed"].unique()):
                df_seed = df_concept[df_concept["seed"] == seed]
                acc_values.append(df_seed["test_acc"].iloc[0])
                alignment_values.append(compute_alignment(df_seed, concept))

            acc_mean.append(float(np.mean(acc_values)))
            acc_std.append(float(np.std(acc_values)))
            alignment_mean.append(float(np.mean(alignment_values)))
            alignment_std.append(float(np.std(alignment_values)))

        data[model] = {
            "concepts": list(CONCEPTS),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "alignment_mean": alignment_mean,
            "alignment_std": alignment_std,
            "rel": RELEVANCY[model],
        }
    return data


def draw_bars(ax1, ax2, model_data):
    concepts = model_data["concepts"]
    x = np.arange(len(concepts)) * GROUP_SPACING

    for spine in ["top", "left", "right", "bottom"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    acc_bars = ax1.bar(
        x - BAR_WIDTH / 2, model_data["acc_mean"], BAR_WIDTH,
        color=ACC_COLOR, linewidth=0,
        # yerr=[np.zeros_like(model_data["acc_std"]), model_data["acc_std"]], error_kw=ERROR_KW,
    )
    alignment_bars = ax2.bar(
        x + BAR_WIDTH / 2, model_data["alignment_mean"], BAR_WIDTH,
        color=ALIGN_COLOR, linewidth=0,
        # yerr=[np.zeros_like(model_data["alignment_std"]), model_data["alignment_std"]], error_kw=ERROR_KW,
    )

    # for bars in (acc_bars, alignment_bars):
    #     lower_caps, upper_caps = bars.errorbar[1]
    #     lower_caps.set_visible(False)

    for j in range(len(concepts)):
        if concepts[j] in model_data["rel"]:
            for ax, bar in [(ax1, acc_bars[j]), (ax2, alignment_bars[j])]:
                ax.add_patch(Rectangle(
                    (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                    fill=False, hatch="/", edgecolor="black", linewidth=0,
                ))

    ax1.set_xticks(x)
    ax1.set_xticklabels(LETTERS)
    ax1.set_ylim(0.5, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax1.tick_params(axis="x", length=0, pad=6)


def draw_legend(legend_ax, x_anchor, relevancy_y):
    legend_ax.axis("off")

    metric_handles = [
        Patch(facecolor=ACC_COLOR, label="Accuracy"),
        Patch(facecolor=ALIGN_COLOR, label="Alignment"),
    ]

    concept_handles = []
    for i in range(len(CONCEPTS)):
        label = f"{LETTERS[i]} - {CONCEPT_NAMES[CONCEPTS[i]]}"
        concept_handles.append(Patch(facecolor="none", edgecolor="none", label=label))

    relevancy_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="///", linewidth=0.5, label="Relevant"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.5, label="Non-Relevant"),
    ]

    metric_legend = legend_ax.legend(
        handles=metric_handles, loc="upper left", bbox_to_anchor=(x_anchor, 1.15),
        frameon=False, fontsize=15, labelspacing=0.15,
    )
    legend_ax.add_artist(metric_legend)

    concept_legend = legend_ax.legend(
        handles=concept_handles, loc="upper left", bbox_to_anchor=(x_anchor, 0.99),
        frameon=False, fontsize=15, handlelength=0, handletextpad=0, labelspacing=0.15,
    )
    legend_ax.add_artist(concept_legend)

    legend_ax.legend(
        handles=relevancy_handles, loc="upper left", bbox_to_anchor=(x_anchor, relevancy_y),
        frameon=False, fontsize=15, handlelength=1.4, handletextpad=0.6, labelspacing=0.15,
    )


def plot_all(data):
    models = list(data.keys())
    n = len(models)

    fig = plt.figure(figsize=(13, 3.4))
    gs = fig.add_gridspec(1, n + 2, width_ratios=[1] * n + [0.05, 0.42], wspace=0.03)

    for i in range(n):
        model = models[i]
        ax1 = fig.add_subplot(gs[0, i])
        ax2 = ax1.twinx()
        ax1.set_title(MODEL_NAMES[model], size=16, pad=5)

        draw_bars(ax1, ax2, data[model])

        if i == 0:
            ax1.tick_params(axis="y", left=True, labelleft=True, colors=ACC_COLOR, length=0, pad=-8)
        else:
            ax1.tick_params(axis="y", left=False, labelleft=False)

        if i == n - 1:
            ax2.tick_params(axis="y", right=True, labelright=True, colors=ALIGN_COLOR, length=0, pad=-8)
        else:
            ax2.tick_params(axis="y", right=False, labelright=False)

    draw_legend(fig.add_subplot(gs[0, n + 1]), x_anchor=-0.1, relevancy_y=0.12)

    plt.subplots_adjust(top=0.925, bottom=0.082, left=0.015, right=0.960)
    plt.show()


def plot_each(data):
    for model in data:
        fig = plt.figure(figsize=(7.8, 3.3))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 0.19], wspace=0.02)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = ax1.twinx()
        ax1.set_title(MODEL_NAMES[model], size=15)

        draw_bars(ax1, ax2, data[model])

        ax1.tick_params(axis="y", left=True, labelleft=True, colors=ACC_COLOR, length=0, pad=-8)
        ax2.tick_params(axis="y", right=True, labelright=True, colors=ALIGN_COLOR, length=0, pad=-8)

        draw_legend(fig.add_subplot(gs[0, 2]), x_anchor=-0.2, relevancy_y=0.11)

        plt.subplots_adjust(top=0.925, bottom=0.085, left=0.020, right=0.880)
        plt.show()


def analysis_rel_nonrel(data):
    results = {}
    for model in data:
        concepts = data[model]["concepts"]
        acc = np.array(data[model]["acc_mean"])
        alignment = np.array(data[model]["alignment_mean"])
        score = (2 * acc - 1) - alignment

        rel_mask = np.array([concept in data[model]["rel"] for concept in concepts])
        nonrel_mask = ~rel_mask

        acc_rel = acc[rel_mask]
        acc_nonrel = acc[nonrel_mask]
        alignment_rel = alignment[rel_mask]
        alignment_nonrel = alignment[nonrel_mask]
        score_rel = score[rel_mask]
        score_nonrel = score[nonrel_mask]

        result = {
            "accuracy_rel": float(np.mean(acc_rel)),
            "accuracy_nonrel": float(np.mean(acc_nonrel)),
            "norm_accuracy_rel": 2 * float(np.mean(acc_rel)) - 1,
            "norm_accuracy_nonrel": 2 * float(np.mean(acc_nonrel)) - 1,
            "alignment_rel": float(np.mean(alignment_rel)),
            "alignment_nonrel": float(np.mean(alignment_nonrel)),
            "score_rel": float(np.mean(score_rel)),
            "score_nonrel": float(np.mean(score_nonrel)),
            "accuracy_diff_rel_minus_nonrel": float(np.mean(acc_rel) - np.mean(acc_nonrel)),
            "alignment_diff_rel_minus_nonrel": float(np.mean(alignment_rel) - np.mean(alignment_nonrel)),
            "std_accuracy_rel": float(np.std(acc_rel)),
            "std_accuracy_nonrel": float(np.std(acc_nonrel)),
            "std_alignment_rel": float(np.std(alignment_rel)),
            "std_alignment_nonrel": float(np.std(alignment_nonrel)),
            "std_score_rel": float(np.std(score_rel)),
            "std_score_nonrel": float(np.std(score_nonrel)),
            "std_accuracy_diff_rel_minus_nonrel": float(np.std(acc_rel) - np.std(acc_nonrel)),
            "std_alignment_diff_rel_minus_nonrel": float(np.std(alignment_rel) - np.std(alignment_nonrel)),
        }
        result["accuracy_all_mean"] = (result["accuracy_rel"] + result["accuracy_nonrel"]) / 2
        result["accuracy_all_std"] = (result["std_accuracy_rel"] + result["std_accuracy_nonrel"]) / 2
        result["alignment_all_mean"] = (result["alignment_rel"] + result["alignment_nonrel"]) / 2
        result["alignment_all_std"] = (result["std_alignment_rel"] + result["std_alignment_nonrel"]) / 2

        results[model] = result
    return results


def main():
    mpl.rcParams["hatch.linewidth"] = 0.3

    df = pd.read_csv(INPUT_CSV)
    df = df[df["probe"] == "linear"].copy()
    df["sensitivity"] = df["sensitivity"].abs()
    df["independence"] = 1 - df["sensitivity"]

    data = summarize(df)
    pprint.pprint(data, indent=4)

    styling()
    plot_all(data)
    plot_each(data)

    results = analysis_rel_nonrel(data)
    for model in results:
        print()
        print(model)
        for metric in results[model]:
            print(metric, results[model][metric])


if __name__ == "__main__":
    main()