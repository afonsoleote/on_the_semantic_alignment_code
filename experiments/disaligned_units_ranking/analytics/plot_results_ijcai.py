import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_CSV = "results_processed.csv"

CORRS = [0.95]
RANKING1 = "replace_sweep_10_full"
RANKING2 = "interp_sweep_10_full"

concept_fmt = {
    "PassengerCar": r"\operatorname{circle\_cargo}",
    "ReinforcedCar": r"\operatorname{reinforced\_car}",
    "FreightWagon": r"\operatorname{freight\_cargo}",
    "OpenRoofCar": r"\operatorname{openroof\_car}",
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

def plot_alignment(df, show_std=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 2.4))

    df = df.sort_values("ranking_to_extract")
    x = df["ranking_to_extract"]

    pc = concept_fmt["PassengerCar"]
    rc = concept_fmt["ReinforcedCar"]

    series = [
        ("test_acc", "Accuracy"),
        ("pc", rf"$\sigma(\neg (\mathsf{{C}} \perp\!\!\!\perp {pc}))$"),
        ("rc", rf"$\sigma(\mathsf{{C}} \perp\!\!\!\perp {rc})$"),
        ("alignment", rf"$\sigma(\neg (\mathsf{{C}} \perp\!\!\!\perp {pc}) \land "
                      rf"(\mathsf{{C}} \perp\!\!\!\perp {rc}))$"),
    ]

    for prefix, label in series:
        mean, std = df[f"{prefix}_mean"], df[f"{prefix}_std"]
        ax.plot(x, mean, label=label)
        if show_std:
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Unit Set", size=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_alignment_subplots(df1, df2, title1, title2, show_std=False):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)

    plot_alignment(df1, show_std=show_std, ax=axes[0])
    plot_alignment(df2, show_std=show_std, ax=axes[1])

    axes[0].set_title(title1, size=12)
    axes[1].set_title(title2, size=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.52, -0.01), handlelength=1.3)

    for ax in axes:
        ax.set_xticks(range(1, 12), [f"$U_{{{i}}}$" for i in range(1, 12)])
        ax.tick_params(axis='x', labelsize=10)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylim((0,1))




    plt.subplots_adjust(top=0.913, bottom=0.360, left=0.050, right=0.990, wspace=0.080)
    return fig, axes


def select(df, corr, ranking):
    return df[df["ranking"] == ranking]


def main():
    styling()
    # styling_ijcai()
    df = pd.read_csv(INPUT_CSV)

    plot_alignment_subplots(
        df[df["ranking"] == RANKING1],
        df[df["ranking"] == RANKING2],
        title1="Procedure 1 -- Set Interpolation",
        title2="Procedure 2 -- Score Interpolation",
        show_std=True,
    )
    plt.show()


if __name__ == "__main__":
    main()