import time

import numpy as np

from experiments.sample_size_stability.config import get_all_configs
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.generation.xtrains.utils import filter_configs
from src.reliability.metrics import causal_sensitivity, counterfactual_accuracy, causal_sensitivity_abs
from src.utils.common_paths import DATASETS_PATH
from src.utils.meta_run import save_experiment, save_summary, load_or_train_probe, runner
from src.utils.pickle import read_configs


def set_data(exp_config):
    dm = ProbingDataModuleXTRAINS([exp_config['out_concept']],
                           'image_id',
                                  exp_config['batch_size'],
                                  exp_config['dataset_cfg'].labels_path,
                                  exp_config['dataset_cfg'].images_path,
                                  exp_config['device'],
                                  exp_config['seed'],
                                  exp_config['model_cfg'].path,
                                  exp_config['model_cfg'].layers)

    dm.set_default_splits(exp_config['tr_size'],
                          exp_config['val_size'],
                          exp_config['test_size'])
    return dm

def main(exp_config):

    ts = time.time()
    print(f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)
    probe, train_results = load_or_train_probe(dm, exp_config)
    exp_config.update(train_results)

    # prepare necessary data for intervention
    train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
                                           datasets_path=str(DATASETS_PATH),
                                           as_dict=True)

    # "invariance"
    if exp_config['out_concept'] != exp_config['protected_concept']:
        _file_names = dm.get_filenames_condition(exp_config['out_concept'], exp_config['protected_concept'])
        if _file_names.empty:
            print(f"Impossible combination of concepts: Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}")
            save_path = save_experiment(exp_config)
            print(f"[INFO] Saved to {save_path}")
            return

    # "causal sensitivity"
    else: # exp_config['out_concept'] == exp_config['protected']
        _file_names = dm.get_filenames(dm.test, only_positive=True)

    _configs = filter_configs(train_configs, _file_names)

    intervention = exp_config['intervention_manager'].from_config(exp_config)

    x, x_cf = dm.get_activations_counterfactual_pairs(_file_names, _configs, intervention)

    # probe default and counterfactual predictions
    preds = probe.predict_proba(x) # [:, 1] # (probe.predict(x) > 0.5).astype(int)
    preds_cf = probe.predict_proba(x_cf) #[:, 1]#(probe.predict(x_cf) > 0.5).astype

    y_gt = np.ones(len(preds))
    y_gt_i = np.zeros(len(preds_cf))

    # grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
    grid = [i for i in range(1,501)]
    total_size = len(preds)
    grid = [n for n in grid if n <= total_size] # drop not possible sizes (almost never)
    rng = np.random.default_rng(exp_config["seed"])

    sens_list = []
    sens_abs_list = []
    indep_list = []
    indep_abs_list = []
    cf_acc_list = []
    for n in grid:
        sel = rng.permutation(total_size)[:n]
        p, pcf = preds[sel], preds_cf[sel]
        yg, ygi = y_gt[sel], y_gt_i[sel]

        sens = causal_sensitivity(p, pcf, yg)
        sens_abs = causal_sensitivity_abs(p, pcf, yg)
        cf_acc = counterfactual_accuracy(p, pcf, yg, ygi)

        sens_list.append(float(sens))
        sens_abs_list.append(float(sens_abs))
        indep_list.append(float(1 - abs(sens)))
        indep_abs_list.append(float(1 - abs(sens_abs)))
        cf_acc_list.append(round(float(cf_acc), 4))

    exp_config['grid'] = grid
    exp_config['sensitivity'] = sens_list
    exp_config['sensitivity_abs'] = sens_abs_list
    exp_config['independence'] = indep_list
    exp_config['independence_abs'] = indep_abs_list
    exp_config['cf_acc'] = cf_acc_list

    del _file_names, _configs, train_configs
    exp_config["time"] = round(time.time() - ts, 3)
    save_path = save_experiment(exp_config)
    print(f"[INFO] Saved to {save_path}")
    print(f"[INFO] Experiment {exp_config['id']} time: {exp_config['time']} s")


if __name__ == '__main__':
    exp_cfgs = get_all_configs()
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])



