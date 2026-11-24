import pandas as pd

import matplotlib.pyplot as plt

from src.generation.xtrains.interventions import intervene_single
from src.utils.common_paths import XTRAINS_NEW_CONFIG_PATH, XTRAINS_NEW_CSV_PATH
from src.utils.pickle import read_configs
from src.generation.xtrains.utils import render_in_mem, show_img


def _check_img_intervention(cfgs):
    imgs = []
    for c in cfgs:
        _c = intervene_single(c, "cargo_type", "circle", sample_values=["none"])
        _c['render_seed'] = 1
        img = render_in_mem(_c)

        imgs.append(img)
        show_img(img)


def _check_number_of_samples_per_concept(df, percentage=False):
    non_hot = ["name", "image_id"]
    for c in df.columns:
        if c not in non_hot:
            if percentage:
                print(c, sum(df[c])/len(df[c]))
            else:
                print(c, sum(df[c]))

def _check_configs(cfgs, amount=10):
    for c in cfgs[:amount]:
        print(c)

def _compute_correlations(df, concepts=None, method="pearson"):

    df = df[concepts]

    df = df[concepts].select_dtypes(include="number")

    corr_matrix = df.corr(method=method)

    plt.figure()
    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)

    cols = corr_matrix.columns
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)

    # add correlation values in the squares
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr_matrix.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    configs = read_configs(XTRAINS_NEW_CONFIG_PATH)
    _df = pd.read_csv(XTRAINS_NEW_CSV_PATH)
    # for c in _df.columns:
    #    print(c)
    # _compute_correlations(_df, concepts=["MixedTrain", "FreightWagon", "EmptyWagon", "PassengerCar", "EmptyTrain", "notEmptyTrain"])#concepts=["2LongWagon", "3Wagon", "LongTrain", "FreightTrain", "LongFreightTrain",] )
    #_check_number_of_samples_per_concept(_df, percentage=True)
    total = len(_df)
    res1 = sum(_df[_df["PassengerTrain"] == 1]["PassengerTrain"])
    res2 = sum(_df[(_df["PassengerLongWagon"] == 0) & (_df["2PassengerCar"]== 1)]["PassengerTrain"])
    res3 = sum(_df[(_df["PassengerLongWagon"] == 1) & (_df["2PassengerCar"]== 0)]["PassengerTrain"])
    res4 = sum(_df[(_df["PassengerLongWagon"] == 1) & (_df["2PassengerCar"]== 1)]["PassengerTrain"])

    print(res1)
    _a = res2/res1
    _b = res3/res1
    _c = res4/res1
    print(_a)
    print(_b)
    print(_c)
    print(_a+_b+_c)
    print()


    total = len(_df)
    res1 = sum(_df[_df["LongTrain"] == 1]["LongTrain"])
    res2 = sum(_df[(_df["3Wagon"] == 0) & (_df["2LongWagon"]== 1)]["LongTrain"])
    res3 = sum(_df[(_df["3Wagon"] == 1) & (_df["2LongWagon"]== 0)]["LongTrain"])
    res4 = sum(_df[(_df["3Wagon"] == 1) & (_df["2LongWagon"]== 1)]["LongTrain"])

    print(res1)
    _a = res2/res1
    _b = res3/res1
    _c = res4/res1
    print(_a)
    print(_b)
    print(_c)
    print(_a+_b+_c)
    print()




