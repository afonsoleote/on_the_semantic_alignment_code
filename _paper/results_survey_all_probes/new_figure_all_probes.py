import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import ScaledTranslation

INPUT_CSV = "3_scatter_acc_alg.csv"

# Panel order follows this dict; models in the CSV but not here go at the end.
MODEL_NAMES = {
    "type_a": r"$f_A$",
    "type_b": r"$f_B$",
    "type_c": r"$f_C$",
    "class_id": r"$f_{GTSRB}$",
    "species": r"$f_{CUB}$",
    "attractive": r"$f_{CelebA}$",
    "income": r"$f_{Adult}$",
}

CONCEPT_NAMES = {
    # XTRAINS
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

    # GTSRB
    'Symbol': r"$\mathsf{\exists hasSymbol.\top}$",
    'Black_Symbol': r"$\mathsf{\exists hasSymbol.Black}$",
    'Black_Bar': r"$\mathsf{\exists hasBar.Black}$",
    'Post': r"$\mathsf{Pole}$",
    'Red_Ground': r"$\mathsf{\exists hasGround.Red}$",
    'Symbol_Speed80': r"$\mathsf{\exists hasSymbol.Speed80}$",
    'White_Ground': r"$\mathsf{\exists hasGround.White}$",

    # CUB
    'attr_4': r"$\mathsf{\exists hasBillShape.Needle}$",
    'attr_5': r"$\mathsf{\exists hasBillShape.Hooked}$",
    'attr_104': r"$\mathsf{\exists hasHeadPattern.Striped}$",
    'attr_112': r"$\mathsf{\exists hasBreastColor.Yellow}$",
    'attr_134': r"$\mathsf{\exists hasThroatColor.Red}$",
    'attr_148': r"$\mathsf{\exists hasEyeColor.Red}$",
    'attr_198': r"$\mathsf{\exists hasBellyColor.Blue}$",
    'attr_204': r"$\mathsf{\exists hasBellyColor.Yellow}$",
    'attr_226': r"$\mathsf{\exists hasShape.DuckLike}$",
    'attr_306': r"$\mathsf{\exists hasCrownColor.White}$",
    'attr_307': r"$\mathsf{\exists hasCrownColor.Red}$",

    # Celeba
    'Blond_Hair': r"$\mathsf{\exists hasHairColor.Blond}$",
    'Brown_Hair': r"$\mathsf{\exists hasHairColor.Brown}$",
    'Black_Hair': r"$\mathsf{\exists hasHairColor.Black}$",
    'Bald': r"$\mathsf{\exists hasHair.\top}$",
    'Bangs': r"$\mathsf{\exists hasHairStyle.Bangs}$",

    'Mustache': r"$\mathsf{\exists hasBeard.Mustache}$",
    'No_Beard': r"$\mathsf{\exists hasBeard.\top}$",

    'Mouth_Slightly_Open': r"$\mathsf{\exists has.MouthOpen}$",

    'Male': r"$\mathsf{Male}$",
    'Young': r"$\mathsf{Young}$",
    'Pale_Skin': r"$\mathsf{\exists hasSkinTone.Pale}$",

    'Bushy_Eyebrows': r"$\mathsf{\exists hasEyebrow.Bushy}$",
    'Eyeglasses': r"$\mathsf{\exists has.Eyeglasses}$",

    # Adult
    'c_advanced_credential': r'$\mathsf{AdvancedCredential}$',
    'c_investment_activity': r'$\mathsf{InvestmentActivity}$',
    'c_prime_working': r'$\mathsf{CareerPeak}$',
    'c_married_man': r'$\mathsf{MarriedMan}$',
    'c_demographic_majority': r'$\mathsf{DemographicMajority}$',
    'c_is_male': r'$\mathsf{Male}$',
    'c_country_early': r'$\mathsf{LowLexicalCountry}$',
    'c_odd_age': r'$\mathsf{OddAge}$',
    'c_workclass_midlen': r'$\mathsf{LongWorkClassName}$',
}

ACC_COLOR = sns.color_palette("deep")[0]
ALIGN_COLOR = sns.color_palette("deep")[1]
PANEL_BG = "none" # "#eaeae6"            # light gray panel background
BAR_WIDTH = 0.4                 # was 0.4; thinner bars, more gap between pairs
GROUP_SPACING = 1.0
ERROR_KW = dict(lw=0.15, capsize=1.0, capthick=0.15, ecolor="black")  # hairline; ~thinnest that still prints
CONCEPT_LABEL_SIZE = 7          # x labels; 65 rotated labels need a small font
LABEL_ROTATION = 50             # degrees; 90 = vertical, 45 = diagonal

# y ranges with a small top margin. Accuracy spans half the range of alignment,
# so its margin is half as big: both "1.0" ticks stay at the same height.
ACC_YLIM = (0.5, 1.01)
ALIGN_YLIM = (0.0, 1.02)

TICK_FMT = "%.2f"               # y tick numbers: 0.00, 0.50, 1.00
BOTTOM_TICK_UP_PT = 3           # lift the lowest tick number (0.50 / 0.00) by this many points

SHOW_GRID = False                # background grid on/off
GRID_COLOR = "white"
GRID_LW = 0.4                   # thin
GRID_LEVELS = 5                 # evenly spaced lines from bottom to 1.0 (top line at 1.0 is skipped)

# letter-mode settings (plot_all_letters)
LETTER_SIZE = 9                 # size of a, b, c under the bars
KEY_SIZE = 7                   # size of "a -- concept name" lines
KEY_LINE_PT = 7               # vertical distance between key lines, in points
# Default position of the key ("a -- name" block) under every panel:
KEY_X_FRAC = 0.06               # x: where the key starts, fraction of the panel width
                                #    (0 = panel's left edge, 0.5 = middle, can be < 0 or > 1)
KEY_TOP_PT = 14                 # y: distance below the panel's bottom edge, in points
                                #    (bigger = lower)

# Per-panel position, overriding the defaults above: model -> (x_frac, y_pt).
# Leave a panel out to use the defaults; use None to keep one of the two defaults,
# e.g. "income": (None, 20) only moves f_Adult's key down.
KEY_POS = {
    "type_a":     (0.03, 14),
    "type_b":     (0.03, 14),
    "type_c":     (0.06, 14),
    "class_id":   (-0.1, 14),
    "species":    (0.08, 14),
    "attractive": (0.08, 14),
    "income":     (0.06, 14),
}


def styling():
    sns.set_theme(context="paper", style="ticks", palette="deep",
                  font_scale=1, color_codes=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        # Times for text and math (mathptmx ships with every LaTeX install;
        # \usepackage{newtxtext,newtxmath} is a nicer alternative if you have it)
        "text.latex.preamble": r"\usepackage{mathptmx}",
        "font.size": 13,
        "axes.linewidth": 0.5,
        "axes.labelsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 9,          # was 15
        "xtick.major.width": 0.2,
        "xtick.major.size": 3,
        "ytick.major.width": 0.2,
        "ytick.major.size": 3,
        "lines.linewidth": 0.8,
    })


def tex_escape(s):
    return s.replace("_", r"\_")


def concept_label(c):
    return CONCEPT_NAMES.get(c, tex_escape(c))


def model_title(m):
    return MODEL_NAMES.get(m, tex_escape(m))


def summarize(df):
    """One entry per model; concepts keep the exact row order of the CSV.

    Uses the accuracy_std / alignment_std columns from the CSV for the error bars.
    """
    df = df.rename(columns=lambda c: c.replace(" ", ""))   # tolerate "alignment _std"
    df[["accuracy_std", "alignment_std"]] = df[["accuracy_std", "alignment_std"]].fillna(0.0)

    present = list(dict.fromkeys(df["model_cfg"]))          # order of first appearance
    order = [m for m in MODEL_NAMES if m in present] + [m for m in present if m not in MODEL_NAMES]

    data = {}
    for model in order:
        sub = df[df["model_cfg"] == model]                  # boolean mask keeps row order
        data[model] = {
            "concepts": sub["out_concept"].tolist(),
            "acc_mean": sub["accuracy"].to_numpy(),
            "acc_std": sub["accuracy_std"].to_numpy(),
            "alignment_mean": sub["alignment"].to_numpy(),
            "alignment_std": sub["alignment_std"].to_numpy(),
        }
    return data


def draw_bars(ax1, ax2, model_data):
    """ax1: accuracy (left scale 0.5-1), ax2: alignment (right scale 0-1).
    Draws bars, error bars and y ranges; x labels are set by the caller."""
    concepts = model_data["concepts"]
    x = np.arange(len(concepts)) * GROUP_SPACING

    for spine in ["top", "left", "right", "bottom"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.set_facecolor(PANEL_BG)

    # symmetric error bars: mean ± std
    ax1.bar(
        x - BAR_WIDTH / 2, model_data["acc_mean"], BAR_WIDTH,
        color=ACC_COLOR, linewidth=0, zorder=3,
        yerr=model_data["acc_std"], error_kw=ERROR_KW,
    )
    ax2.bar(
        x + BAR_WIDTH / 2, model_data["alignment_mean"], BAR_WIDTH,
        color=ALIGN_COLOR, linewidth=0, zorder=3,
        yerr=model_data["alignment_std"], error_kw=ERROR_KW,
    )

    ax1.set_xticks(x)
    ax1.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax1.set_ylim(*ACC_YLIM)     # accuracy: chance level at the bottom
    ax2.set_ylim(*ALIGN_YLIM)   # alignment
    ax1.set_yticks([0.5, 0.75, 1.0])
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax1.yaxis.set_major_formatter(FormatStrFormatter(TICK_FMT))
    ax2.yaxis.set_major_formatter(FormatStrFormatter(TICK_FMT))

    if SHOW_GRID:
        # thin white lines behind the bars (drawn on the accuracy axis; on the
        # alignment scale they fall at the same relative heights)
        for y in np.linspace(ACC_YLIM[0], 1.0, GRID_LEVELS)[:-1]:
            ax1.axhline(y, color=GRID_COLOR, linewidth=GRID_LW, zorder=0.5)

    ax1.tick_params(axis="x", length=0, pad=3)
    return concepts


def label_with_names(ax1, concepts, model=None):
    """x labels = full concept names, rotated."""
    ax1.set_xticklabels([concept_label(c) for c in concepts],
                        rotation=LABEL_ROTATION, ha="right", rotation_mode="anchor",
                        fontsize=CONCEPT_LABEL_SIZE)


def label_with_letters(ax1, concepts, model=None):
    """x labels = a, b, c, ...; below the panel a key 'a -- name' per line.
    Letters restart at 'a' in every panel. The key is left-aligned; its position
    is KEY_POS[model] if given, otherwise (KEY_X_FRAC, KEY_TOP_PT)."""
    letters = [chr(ord("a") + j) for j in range(len(concepts))]
    ax1.set_xticklabels(letters, rotation=0, fontsize=LETTER_SIZE)

    x_frac, y_pt = KEY_POS.get(model, (None, None))
    x_frac = KEY_X_FRAC if x_frac is None else x_frac
    y_pt = KEY_TOP_PT if y_pt is None else y_pt
    key = "\n".join(rf"{letter} -- {concept_label(c)}" for letter, c in zip(letters, concepts))
    ax1.annotate(
        key,
        xy=(x_frac, 0), xycoords="axes fraction",
        xytext=(0, -y_pt), textcoords="offset points",
        ha="left", va="top", multialignment="left",
        linespacing=KEY_LINE_PT / KEY_SIZE, fontsize=KEY_SIZE, annotation_clip=False,
    )


def draw_legend(fig, loc="lower center", anchor=(0.5, -0.05), ncol=2, fontsize=15):
    """anchor = (x, y) in figure fractions: (0, 0) bottom-left, (1, 1) top-right.
    loc says which corner/side of the legend sits on the anchor point."""
    metric_handles = [
        Patch(facecolor=ACC_COLOR, label="Accuracy"),
        Patch(facecolor=ALIGN_COLOR, label="Alignment"),
    ]
    fig.legend(
        handles=metric_handles, loc=loc, bbox_to_anchor=anchor,
        ncol=ncol, frameon=False, fontsize=fontsize, handlelength=1.2, columnspacing=1.5,
    )


def make_panels(data, figsize, label_fn, legend_kw=None):
    """Shared figure builder: one panel per model, twin y scales, legend."""
    models = list(data.keys())
    sizes = [len(data[m]["concepts"]) for m in models]

    fig, axes = plt.subplots(
        1, len(models), figsize=figsize,
        gridspec_kw=dict(width_ratios=sizes),   # same bar width in every panel
    )
    axes = np.atleast_1d(axes)
    n = len(models)

    for i, (ax1, model) in enumerate(zip(axes, models)):
        ax2 = ax1.twinx()
        ax1.set_title(model_title(model), size=12, pad=5)
        concepts = draw_bars(ax1, ax2, data[model])
        label_fn(ax1, concepts, model)

        # left scale (accuracy, blue) only on the first panel
        if i == 0:
            ax1.tick_params(axis="y", left=True, labelleft=True, colors=ACC_COLOR, length=0, pad=2)
        else:
            ax1.tick_params(axis="y", left=False, labelleft=False)

        # right scale (alignment, orange) only on the last panel
        if i == n - 1:
            last_ax2 = ax2
            ax2.tick_params(axis="y", right=True, labelright=True, colors=ALIGN_COLOR, length=0, pad=2)
        else:
            ax2.tick_params(axis="y", right=False, labelright=False)

    # lift the lowest y number on the left (first panel) and right (last panel)
    lift = ScaledTranslation(0, BOTTOM_TICK_UP_PT / 72, fig.dpi_scale_trans)
    for ax in (axes[0], last_ax2):
        lowest = ax.get_yticklabels()[0]
        lowest.set_transform(lowest.get_transform() + lift)

    draw_legend(fig, **(legend_kw or {}))
    return fig


def plot_all(data):
    """Concept names written under the bars (rotated)."""
    make_panels(data, figsize=(9, 2.2), label_fn=label_with_names, legend_kw=dict(loc="lower center", anchor=(0.5, -0.04), ncol=2, fontsize=10))

    # adjust manually
    plt.subplots_adjust(top=0.900, bottom=0.550, left=0.055,  right=0.970,  hspace=0.200, wspace=0.125)
    plt.savefig("acc_alg_bars.pdf")
    plt.show()


def plot_all_letters(data):
    """Alternative: letters a, b, c under the bars, key 'a -- name' below each panel."""
    # Legend centred below all the keys, in 2 columns.
    # anchor=(x, y) in figure fractions: raise y to move it up.
    make_panels(data, figsize=(8.9, 2.5), label_fn=label_with_letters,
                legend_kw=dict(loc="lower center", anchor=(0.5, -0.04), ncol=2, fontsize=10))

    # adjust manually: bottom leaves room for the longest key (11 lines) + legend;
    # wspace keeps long names from running into the next panel
    plt.subplots_adjust(top=0.920, bottom=0.560, left=0.030,  right=0.955,  hspace=0.200, wspace=0.060)
    plt.savefig("acc_alg_bars_letters.pdf")
    plt.show()


def main():
    df = pd.read_csv(INPUT_CSV)
    data = summarize(df)
    styling()
    plot_all(data)
    plot_all_letters(data)   # alternative layout


if __name__ == "__main__":
    main()