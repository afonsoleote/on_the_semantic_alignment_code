import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from scripts.concept_units_new.config import get_all_configs
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.generation.xtrains.utils import filter_configs
from src.utils.common_paths import DATASETS_PATH
from src.utils.pickle import read_configs


def kde(x, bw_scale=0.5):
    k = gaussian_kde(x)
    k.set_bandwidth(k.factor * bw_scale)
    return k

def kde_sep(p, n, n_grid=1000):
    kp, kn = kde(p), kde(n)
    grid = np.linspace(min(p.min(), n.min()), max(p.max(), n.max()), n_grid)
    ovl = np.trapezoid(np.minimum(kp(grid), kn(grid)), grid)
    return float(1 - ovl)

def get_pos_neg(dm, train_configs):
    fn_pos = dm.get_filenames_subset(dm.test, pos=True)
    fn_neg = dm.get_filenames_subset(dm.test, pos=False)
    img_pos = dm.get_imgs_from_config(fn_pos, filter_configs(train_configs, fn_pos))
    img_neg = dm.get_imgs_from_config(fn_neg, filter_configs(train_configs, fn_neg))
    pos = dm.get_activations_from_imgs(img_pos).detach().cpu().numpy()
    neg = dm.get_activations_from_imgs(img_neg).detach().cpu().numpy()
    return pos.ravel(), neg.ravel()


def main(exp_config, plot=True):
    concept = exp_config['out_concept']
    dm = ProbingDataModuleXTRAINS([concept], 'image_id', exp_config['batch_size'],
                                  exp_config['dataset_cfg'].labels_path,
                                  exp_config['dataset_cfg'].images_path,
                                  exp_config['device'], exp_config['seed'],
                                  exp_config['model_cfg'].path, exp_config['to_extract'])
    dm.set_default_splits(exp_config['tr_size'], exp_config['val_size'], exp_config['test_size'])

    train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
                                 datasets_path=str(DATASETS_PATH), as_dict=True)

    pos, neg = get_pos_neg(dm, train_configs)
    score =  kde_sep(pos, neg)
    print(f"{concept}: score {score}")

    return {"kde_sep": score}


if __name__ == '__main__':

    exp_configs = get_all_configs()
    rows = []

    for i, exp_cfg in enumerate(exp_configs):
        print(f"({i}/{len(exp_configs)})" + str(exp_cfg['to_extract']))
        res = main(exp_cfg, plot=True)

        row = {}

        row.update(exp_cfg)

        row["to_extract"] = str(exp_cfg["to_extract"])

        row.update(res)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("ranking_full.csv", index=False)