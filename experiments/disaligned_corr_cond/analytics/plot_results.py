import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_CSV = "results_processed.csv"

blue = sns.color_palette("deep")[0]
green_colors = [
    sns.color_palette("deep")[2],
    "#006D2C",
    "#8DD3C1",
]


CURVES = [
    {
        "protected_concept": "PassengerCar",
        "condition": "none",
        "metric": "test_acc",
        "label": "Accuracy",
        "color": blue,
    },

    {
        "protected_concept": "PassengerCar",
        "condition": "none",
        "metric": "semantics",
        "label": r"$\sigma(\neg (\mathsf{C} \indep \operatorname{circle\_cargo}))$",
        "color": green_colors[0],
    },

    {
        "protected_concept": "PassengerCar",
        "condition": "FreightWagon=0",
        "metric": "semantics",
        "label": r"$\sigma(\neg (\mathsf{C} \indep \operatorname{circle\_cargo} \mid \neg \operatorname{freight\_cargo}))$",
        "color": green_colors[2],
    },

    {
        "protected_concept": "PassengerCar",
        "condition": "FreightWagon=1",
        "metric": "semantics",
        "label": r"$\sigma(\neg (\mathsf{C} \indep \operatorname{circle\_cargo} \mid \operatorname{freight\_cargo}))$",
        "color": green_colors[1],
    },

]


def styling():
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font_scale=1, color_codes=True, rc=None)
    plt.rcParams['text.latex.preamble'] = (
        r"\usepackage{amsmath}\newcommand{\indep}{\mathrel{\perp\!\!\!\perp}}"
    )
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.5,
        'axes.labelsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.major.width': 0.2,
        'xtick.major.size': 3,
        'ytick.major.width': 0.2,
        'ytick.major.size': 3,
        'lines.linewidth': 1,
    })


def main():
    styling()
    df = pd.read_csv(INPUT_CSV)

    fig, ax = plt.subplots(figsize=(4.2, 2.4))

    for curve in CURVES:
        rows = df[(df["protected_concept"] == curve["protected_concept"])
                  & (df["condition"] == curve["condition"])
                  & (df["metric"] == curve["metric"])].sort_values("corr")

        x = rows["corr"]
        y = rows["mean"]
        std = rows["std"]

        ax.plot(x, y, color=curve["color"], label=curve["label"])
        ax.fill_between(x, y - std, y + std, color=curve["color"],
                        alpha=0.2, linewidth=0)

    ax.set_xlabel("Correlation")
    ax.set_ylim(0, 1.01)
    ax.set_xlim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    plt.subplots_adjust(top=0.978, bottom=0.168, left=0.07, right=0.992)
    plt.show()


if __name__ == "__main__":
    main()