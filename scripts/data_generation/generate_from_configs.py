import pandas as pd
import math

from src.utils.common_paths import XTRAINS_NEW_CONFIG_PATH, XTRAINS_NEW_CSV_PATH
from src.utils.pickle import read_configs
from src.generation.xtrains.utils import render


def generate_images(cfgs, start=None):
    start = 0 if None else start
    total_samples = len(cfgs)
    for sample_num, c in enumerate(cfgs):
        if sample_num >= start:
            if sample_num % math.ceil(total_samples * 0.05) == 0:
                print("[INFO] Generating example: {} of {}".format(sample_num, total_samples))
            render(c, c["image_id"])



if __name__ == "__main__":
    configs = read_configs(XTRAINS_NEW_CONFIG_PATH)
    df = pd.read_csv(XTRAINS_NEW_CSV_PATH)
    # generate_images(configs, start=30_000)


