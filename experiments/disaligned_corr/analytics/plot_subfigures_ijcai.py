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

subfig_labels = ["(a)", "(b)", "(c)", "(d)"]

def styling_ijcai():
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font_scale=1, color_codes=True, rc=None)
    plt.rcParams['text.latex.preamble'] = (
        r"\usepackage{amsmath}\usepackage{mathptmx}"
        r"\newcommand{\indep}{\mathrel{\perp\!\!\!\perp}}"
        r"\DeclareMathAlphabet{\mathlm}{OT1}{lmr}{m}{n}"
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


def plot_curves(c, ax):
    dep = concept_fmt[c["concept_dep"].iloc[0]]
    indep = concept_fmt[c["concept_indep"].iloc[0]]

    lines = [
        ("acc", r"Accuracy"),
        ("sens", rf"$\sigma(\neg (\mathsf{{C}} \perp\!\!\!\perp {dep}))$"),
        ("indep", rf"$\sigma(\mathsf{{C}}\perp\!\!\!\perp {indep})$"),
        ("alg", rf"$\sigma(\neg (\mathsf{{C}} \perp\!\!\!\perp {dep}) \land "
                rf"(\mathsf{{C}} \perp\!\!\!\perp {indep}))$"),
    ]

    x = c["corr"]
    for df_label, legend_label in lines:
        mean, std = c[f"{df_label}_mean"], c[f"{df_label}_std"]
        ax.plot(x, mean, marker="o", ms=0, label=legend_label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_ylim(0, 1.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_groups(df):
    groups = []
    for pair in order:
        df_group = df[df["corr_pair"] == str(pair)].sort_values("corr")
        print("C:", df_group["out_concept"].iloc[0])
        print("corr:", str(pair))
        print(df_group)
        groups.append(df_group)
    return groups


def plot_grid(groups):
    """2 x 2 layout."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.8), sharex=True, sharey=True)

    for ax, c, sub in zip(axes.flat, groups, subfig_labels):
        plot_curves(c, ax)
        ax.set_title(sub, fontsize=11)
        ax.legend(fontsize=8, frameon=False)

    for ax in axes[1, :]:
        ax.set_xlabel("Correlation")

    fig.tight_layout()
    # plt.savefig("all_pairs_2x2.pdf", bbox_inches="tight")
    plt.show()
    return fig

def plot_grid2(groups, mode="plain"):
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.3), sharey=True)   # narrower: was 7

    for ax, c, sub, pair in zip(axes.flat, groups, subfig_labels, order):
        plot_curves(c, ax)
        ax.set_title(sub, fontsize=11)
        if mode == "pair":
            ax.set_xlabel(rf"$\text{{Corr}}(\mbox{{\small ${concept_fmt[pair[0]]}$}}, \ \mbox{{\small ${concept_fmt[pair[1]]}$}})$")
        else:
            ax.set_xlabel("Correlation")
        ax.legend(fontsize=9,              # was 10
                  labelspacing=0.2,        # space between legend rows (default 0.5)
                  frameon=False, loc="upper left",
                  bbox_to_anchor=(-0.02, 0.51), handlelength=1.3)

    # y ticks: shared by all panels, so setting them once is enough
    axes[0, 0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0, 0].set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    # adjust manually; hspace leaves room for the top row's x labels
    plt.subplots_adjust(top=0.94, bottom=0.10, left=0.045, right=0.99,
                        wspace=0.08, hspace=0.55)
    # plt.savefig("all_pairs_2x2.pdf")

    plt.show()
    return fig

def plot_row(groups, mode="plain"):
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.5), sharey=True)

    for ax, c, sub, pair in zip(axes, groups, subfig_labels, order):
        plot_curves(c, ax)
        ax.set_title(sub, fontsize=11)
        if mode == "pair":
            ax.set_xlabel(rf"$\text{{Corr}}(\mbox{{\small ${concept_fmt[pair[0]]}$}}, \ \mbox{{\small ${concept_fmt[pair[1]]}$}})$")
        else:
            ax.set_xlabel("Correlation")
        ax.legend(fontsize=10, frameon=False, loc="upper left",
                  bbox_to_anchor=(-0.02, 0.51), handlelength=1.3)

    # fig.tight_layout()
    # plt.savefig("all_pairs_1x4.pdf", bbox_inches="tight")
    plt.subplots_adjust(top=0.918, bottom=0.183, left=0.025, right=0.992, wspace=0.100)

    plt.show()
    return fig


def main():
    styling()
    # styling_ijcai()
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(PROCESSED_CSV)

    groups = get_groups(df)
    # plot_grid(groups)
    # plot_row(groups, "pair")
    plot_grid2(groups, "pair")

if __name__ == "__main__":
    main()