import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROCESSED_CSV = "results_processed.csv"

order = [
    ("PassengerCar", "ReinforcedCar"),
    ("PassengerCar", "FreightWagon"),
    ("OpenRoofCar", "PassengerCar"),
    ("PassengerCar", "OpenRoofCar"),
]

concept_fmt = {
    "PassengerCar": r"\operatorname{circle\_cargo}",
    "ReinforcedCar": r"\operatorname{reinforced\_car}",
    "FreightWagon": r"\operatorname{freight\_cargo}",
    "OpenRoofCar": r"\operatorname{open\_roof}",
}


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


def plot_curves(c, lines, fig_id):
    x = c["corr"]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))

    for df_label, legend_label in lines:
        mean, std = c[f"{df_label}_mean"], c[f"{df_label}_std"]
        ax.plot(x, mean, marker="o", ms=0, label=legend_label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_ylim(0, 1.01)
    # ax.set_xlabel("Correlation")
    ax.set_xlabel(
        rf"$\text{{Corr}}(\mbox{{\small ${concept_fmt[c["concept_dep"].iloc[0]]}$}}, \ \mbox{{\small ${concept_fmt[c["concept_indep"].iloc[0]]}$}})$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.48))
    fig.tight_layout()
    plt.subplots_adjust(top=0.978, bottom=0.193, left=0.07, right=0.992)
    plt.savefig(fig_id, dpi=400)
    # plt.show()
    return fig


def main():
    styling()
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(PROCESSED_CSV)

    for pair in order:
        df_group = df[df["corr_pair"] == str(pair)].sort_values("corr")

        print("C:", df_group["out_concept"].iloc[0])
        print("corr:", str(pair))
        print(df_group)

        dep = concept_fmt[df_group["concept_dep"].iloc[0]]
        indep = concept_fmt[df_group["concept_indep"].iloc[0]]

        dep_no_fmt = df_group["concept_dep"].iloc[0]
        indep_no_fmt = df_group["concept_indep"].iloc[0]

        lines = [
            ("acc", r"Accuracy"),
            ("sens", rf"$\sigma(\neg (\mathsf{{C}} \perp\!\!\!\perp {dep}))$"),
            ("indep", rf"$\sigma(\mathsf{{C}}\perp\!\!\!\perp {indep})$"),
            ("alg", rf"$\sigma(\neg (\mathsf{{C}} \perp\!\!\!\perp {dep}) \land "
                    rf"(\mathsf{{C}} \perp\!\!\!\perp {indep}))$"),
        ]
        for i in range(len(lines)):
            # plot_curves(df_group, lines[0:i+1], f"./figures_slides/{dep_no_fmt}_{indep_no_fmt}_{i}.pdf")
            plot_curves(df_group, lines[0:i+1], f"./figures_slides/{dep_no_fmt}_{indep_no_fmt}_{i}.png")


if __name__ == "__main__":
    main()