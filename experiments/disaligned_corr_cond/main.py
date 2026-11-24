import time

import numpy as np
import pandas as pd

from experiments.disaligned_corr_cond.config import get_all_configs
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.generation.xtrains.utils import filter_configs
from src.reliability.metrics import causal_sensitivity, counterfactual_accuracy, causal_sensitivity_abs
from src.utils.common_paths import DATASETS_PATH
from src.utils.meta_run import save_experiment, save_summary, load_or_train_probe, runner
from src.utils.pickle import read_configs


def make_split(dm, attr_a, attr_b, attr_r, rho, seed,
               n_train=800, n_val=200, n_test=1000,
               p_r_train=0.9):

    p_r_train = rho
    rng = np.random.default_rng(seed)

    # indexes for each split.
    samples = {
        "train": [],
        "val": [],
        "test": [],
        "test_alg": []
    }

    split_names = ["train", "val", "test", "test_alg"]
    split_sizes = [n_train, n_val, n_test, n_test]
    r_probabilities = [p_r_train, p_r_train, p_r_train, 0.5]

    for i in range(len(split_names)):
        split = split_names[i]
        n = split_sizes[i]
        p_r = r_probabilities[i]

        # each value of r defines a region built separetely
        for r in [0, 1]:
            p_region = p_r if r == 1 else 1 - p_r
            n_region = round(n * p_region)
            n_per_a = n_region // 2

            # r=1, rho controls the correlation between a and b
            # r=0, a and b are built independently.
            if r == 1:
                n_same = round(n_per_a * (1 + rho) / 2)
            else:
                n_same = n_per_a // 2

            for a in [0, 1]:
                n_other = n_per_a - n_same

                # b = a
                pool = dm.df.index[
                    (dm.df[attr_r] == r)
                    & (dm.df[attr_a] == a)
                    & (dm.df[attr_b] == a)
                ]

                samples[split].extend(
                    rng.choice(pool, n_same, replace=False)
                )

                # b != a
                pool = dm.df.index[
                    (dm.df[attr_r] == r)
                    & (dm.df[attr_a] == a)
                    & (dm.df[attr_b] == 1 - a)
                ]

                samples[split].extend(
                    rng.choice(pool, n_other, replace=False)
                )

    dm.tr = pd.Index(rng.permutation(samples["train"]))
    dm.val = pd.Index(rng.permutation(samples["val"]))
    dm.test = pd.Index(rng.permutation(samples["test"]))
    dm.test_alg = pd.Index(rng.permutation(samples["test_alg"]))


def set_data(exp_config):
    dm = ProbingDataModuleXTRAINS([exp_config['out_concept']],
                           'image_id',
                                  exp_config['batch_size'],
                                  DATASETS_PATH / "xtrains_new/labels_conditionals.csv",
                                  exp_config['dataset_cfg'].images_path,
                                  exp_config['device'],
                                  exp_config['seed'],
                                  exp_config['model_cfg'].path,
                                  exp_config['model_cfg'].layers)

    print("concept", exp_config['out_concept'] )
    print("correlated_concept", exp_config['disalign'][3] )
    print("region", exp_config['region_feats'] )
    print(exp_config['disalign'][2], exp_config['disalign'][3], exp_config['disalign'][1])
    corr = exp_config['disalign'][1]
    attr_a = exp_config['out_concept']
    attr_b = exp_config['disalign'][3]
    attr_r = exp_config['region_feats'][0]
    make_split(dm, attr_a, attr_b, attr_r, corr, seed=exp_config['seed'])
    return dm

def main(exp_config):

    ts = time.time()
    print(f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Conditional: {exp_config["conditional"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)
    probe, train_results = load_or_train_probe(dm, exp_config)
    exp_config.update(train_results)

    # prepare necessary data for intervention
    exp_config["epochs"] = 1000
    train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
                                           datasets_path=str(DATASETS_PATH),
                                           as_dict=True)



    _file_names = dm.get_conditioned_filenames(exp_config["protected_concept"], conditionals=exp_config['conditional'])
    if _file_names.empty:
        print(
            f"Impossible combination of concepts: Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}")
        save_path = save_experiment(exp_config)
        print(f"[INFO] Saved to {save_path}")
        return



    _configs = filter_configs(train_configs, _file_names)

    # get intervention from config
    intervention = exp_config['intervention_manager'].from_config(exp_config)

    x, x_cf = dm.get_activations_counterfactual_pairs(_file_names, _configs, intervention)

    # probe default and counterfactual predictions
    preds = probe.predict_proba(x)
    preds_cf = probe.predict_proba(x_cf)

    y_gt = np.ones(len(preds))
    y_gt_i = np.zeros(len(preds_cf))

    # metrics
    sensitiviy = causal_sensitivity(preds, preds_cf, y_gt)
    sensitiviy_filtered = causal_sensitivity(preds, preds_cf, y_gt, filtered=True)
    sensitiviy_abs = causal_sensitivity_abs(preds, preds_cf, y_gt)
    sensitiviy_abs_filtered = causal_sensitivity_abs(preds, preds_cf, y_gt, filtered=True)

    cf_accuracy = round(counterfactual_accuracy(preds, preds_cf, y_gt, y_gt_i,), 4)

    # store results
    exp_config['sensitivity'] = sensitiviy
    exp_config['sensitivity_f'] = sensitiviy_filtered
    exp_config['sensitivity_abs'] = sensitiviy_abs
    exp_config['sensitivity_abs_f'] = sensitiviy_abs_filtered

    exp_config['independence'] = 1 - abs(sensitiviy)
    exp_config['independence_f'] = 1 - abs(sensitiviy_filtered)
    exp_config['independence_abs'] = 1 - abs(sensitiviy_abs)
    exp_config['independence_abs_f'] = 1 - abs(sensitiviy_abs_filtered)

    exp_config['cf_acc'] = cf_accuracy

    print(f"[INFO] Causal sensitivity (all): {sensitiviy:.2f}")
    print(f"[INFO] Causal sensitivity (abs): {sensitiviy_abs:.2f}")

    del _file_names, _configs, train_configs

    exp_config["time"] = round(time.time() - ts, 3)
    save_path = save_experiment(exp_config)
    print(f"[INFO] Saved to {save_path}")
    print(f"[INFO] Experiment {exp_config["id"]} time: {exp_config["time"]} s")



if __name__ == '__main__':
    ts = time.time()

    exp_cfgs = get_all_configs()
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])

    print(round(time.time() - ts, 3))


