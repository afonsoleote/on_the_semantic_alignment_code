import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INPUT_CSV = "3_scatter_acc_alg.csv"

RAND_ACC = 0.5
PLOT_TOL = 0.15
TOLERANCES = [0.05, 0.10, 0.15, 0.20, 0.25]
ACC_THRESHOLD = 0.90

BLUE = "#4c72b0"
RED = "#c44e52"
GRAY = "0.75"


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
        "lines.linewidth": 1,
    })

def styling_ijcai():
    sns.set_theme(context="paper", style="ticks", palette="deep",
                  font_scale=1, color_codes=True)
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}"
        r"\usepackage{times}"
        r"\newcommand{\indep}{\mathrel{\perp\!\!\!\perp}}"
    )
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 10,
        "axes.linewidth": 0.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 0.2,
        "xtick.major.size": 3,
        "ytick.major.width": 0.2,
        "ytick.major.size": 3,
        "lines.linewidth": 1,
    })



def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(df["accuracy"], df["alignment"], alpha=0.9, s=25)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Alignment")
    ax.set_xlim(0.49, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.125, bottom=0.145, right=0.985, top=0.980, wspace=0.200, hspace=0.200)
    plt.show()




def plot_tolerance(df, tol=PLOT_TOL, tol_max=0.25):
    x = df["accuracy"].to_numpy()
    y = df["alignment"].to_numpy()
    expected = (x - RAND_ACC) / (1.0 - RAND_ACC)
    diff = np.abs(y - expected)
    inside = diff <= tol
    outside = diff > tol

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    ax = axes[0]

    grid = np.linspace(0.49, 1.02, 200)
    line = (grid - RAND_ACC) / (1.0 - RAND_ACC)
    ax.plot(grid, line, ls="--", lw=1.1, color=GRAY, label="Alignment = Accuracy (Rescaled)")
    ax.fill_between(grid, line - tol, line + tol, color=GRAY, alpha=0.20, lw=0)

    ax.scatter(x[inside], y[inside], alpha=0.9, s=25, color=BLUE)
    ax.scatter(x[outside], y[outside], alpha=0.9, s=25, color=RED)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Alignment")
    ax.set_xlim(0.49, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.set_title("(a)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[1]
    tolerances = np.linspace(0, tol_max, 400)
    fractions = []
    for t in tolerances:
        fractions.append(np.mean(diff <= t))
    ax.plot(tolerances, fractions, lw=1.6, color=BLUE)

    ax.set_xlabel("Tolerance")
    ax.set_ylabel(r"$\%$ of Probes")
    ax.set_xlim(0, tol_max + 0.01)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b)", fontsize=12)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.065, bottom=0.145, right=0.980, top=0.930, wspace=0.28)
    plt.show()


def print_statistics(df):
    x = df["accuracy"].to_numpy()
    y = df["alignment"].to_numpy()
    line = (x - RAND_ACC) / (1.0 - RAND_ACC)
    diff = np.abs(y - line)

    print(f"All probes (n = {len(x)})")
    for tol in TOLERANCES:
        inside_percentage = int(round(100 * np.mean(diff <= tol)))
        print(f"  treshold {tol}: inside {inside_percentage}%, outside {100 - inside_percentage}%")
    print()

    filtered = diff[x > ACC_THRESHOLD]

    print(f"Probes with acc > {ACC_THRESHOLD} (n = {len(filtered)})")
    for tol in TOLERANCES:
        inside_pct = int(round(100 * np.mean(filtered <= tol)))
        print(f"  tolerance {tol}: inside {inside_pct}%, outside {100 - inside_pct}%")


def main():
    styling()
    # styling_ijcai()
    df = pd.read_csv(INPUT_CSV)

    plot_scatter(df)
    plot_tolerance(df)
    print_statistics(df)


if __name__ == "__main__":
    main()