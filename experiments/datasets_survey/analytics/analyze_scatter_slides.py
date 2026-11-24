import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter

INPUT_CSV = "3_scatter_acc_alg.csv"
OUTPUT_GIF = "tolerance.gif"

RAND_ACC = 0.5
PLOT_TOL = 0.10
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


def tolerance_schedule(start=PLOT_TOL, low=0.02, high=0.25, step=0.002):
    """start -> low -> high -> start, in steps of `step`, without repeating endpoints.
    Returns the tolerances and the indices of the turning points."""
    n = lambda a, b: int(round(abs(b - a) / step)) + 1
    down = np.linspace(start, low, n(start, low))
    up = np.linspace(low, high, n(low, high))[1:]
    back = np.linspace(high, start, n(high, start))[1:]
    tols = np.round(np.concatenate([down, up, back]), 6)
    i_low = len(down) - 1
    i_high = i_low + len(up)
    return tols, {0, i_low, i_high, len(tols) - 1}


def animate_tolerance(df, out=OUTPUT_GIF, start=PLOT_TOL, tol_min=0.02, tol_max=0.25,
                      step=0.002, fps=25, pause_frames=25, dpi=200):
    x = df["accuracy"].to_numpy()
    y = df["alignment"].to_numpy()
    expected = (x - RAND_ACC) / (1.0 - RAND_ACC)
    diff = np.abs(y - expected)

    tols, turning = tolerance_schedule(start, tol_min, tol_max, step)
    # hold a few frames at the turning points (start, tol_min, tol_max, back to start)
    frames = []
    for i, t in enumerate(tols):
        frames += [t] * (pause_frames if i in turning else 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # ---------------- (a) scatter with tolerance band ----------------
    ax = axes[0]
    grid = np.linspace(0.49, 1.02, 200)
    line = (grid - RAND_ACC) / (1.0 - RAND_ACC)
    ax.plot(grid, line, ls="--", lw=1.1, color=GRAY, label="Alignment = Accuracy (Rescaled)")
    band = [ax.fill_between(grid, line - frames[0], line + frames[0], color=GRAY, alpha=0.20, lw=0)]
    sc = ax.scatter(x, y, alpha=0.9, s=25, color=BLUE)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Alignment")
    ax.set_xlim(0.49, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # ---------------- (b) fraction inside vs tolerance ----------------
    ax = axes[1]
    tolerances = np.linspace(0, tol_max, 400)
    fractions = np.array([np.mean(diff <= t) for t in tolerances])
    ax.plot(tolerances, fractions, lw=1.6, color=BLUE)
    vline, = ax.plot([], [], ls="--", lw=1.0, color="0.4")
    hline, = ax.plot([], [], ls="--", lw=1.0, color="0.4")

    ax.set_xlabel("Tolerance")
    ax.set_ylabel(r"$\%$ of Probes")
    ax.set_xlim(0, tol_max + 0.01)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b)", fontsize=12)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.065, bottom=0.17, right=0.980, top=0.930, wspace=0.28)

    def update(t):
        band[0].remove()
        band[0] = axes[0].fill_between(grid, line - t, line + t, color=GRAY, alpha=0.20, lw=0)

        inside = diff <= t
        sc.set_color(np.where(inside, BLUE, RED))

        frac = inside.mean()
        vline.set_data([t, t], [-0.02, frac])
        hline.set_data([0, t], [frac, frac])
        return band[0], sc, vline, hline

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {out}")


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


def plot_tolerance_snapshot(df, tol=0.10, tol_max=0.25, out=None):
    """Static version of one animation frame: band and colours at `tol`,
    dashed lines on (b) stopping at `tol`. Saves to `out` if given."""
    x = df["accuracy"].to_numpy()
    y = df["alignment"].to_numpy()
    expected = (x - RAND_ACC) / (1.0 - RAND_ACC)
    diff = np.abs(y - expected)
    inside = diff <= tol

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # ---------------- (a) scatter with tolerance band ----------------
    ax = axes[0]
    grid = np.linspace(0.49, 1.02, 200)
    line = (grid - RAND_ACC) / (1.0 - RAND_ACC)
    ax.plot(grid, line, ls="--", lw=1.1, color=GRAY, label="Alignment = Accuracy (Rescaled)")
    ax.fill_between(grid, line - tol, line + tol, color=GRAY, alpha=0.20, lw=0)
    ax.scatter(x, y, alpha=0.9, s=25, color=np.where(inside, BLUE, RED))

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Alignment")
    ax.set_xlim(0.49, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # ---------------- (b) fraction inside vs tolerance ----------------
    ax = axes[1]
    tolerances = np.linspace(0, tol_max, 400)
    fractions = np.array([np.mean(diff <= t) for t in tolerances])
    ax.plot(tolerances, fractions, lw=1.6, color=BLUE)

    frac = inside.mean()
    ax.plot([tol, tol], [-0.02, frac], ls="--", lw=1.0, color="0.4")
    ax.plot([0, tol], [frac, frac], ls="--", lw=1.0, color="0.4")

    ax.set_xlabel("Tolerance")
    ax.set_ylabel(r"$\%$ of Probes")
    ax.set_xlim(0, tol_max + 0.01)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b)", fontsize=12)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.065, bottom=0.17, right=0.980, top=0.930, wspace=0.28)
    if out:
        fig.savefig(out, dpi=200)
    plt.show()


def main():
    styling()
    # styling_ijcai()
    df = pd.read_csv(INPUT_CSV)

    # plot_scatter(df)
    # plot_tolerance(df)
    # animate_tolerance(df)
    # print_statistics(df)
    plot_tolerance_snapshot(df)


if __name__ == "__main__":
    main()