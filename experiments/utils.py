import numpy as np

from src.generation.xtrains.utils import filter_configs
from src.reliability.metrics import causal_sensitivity, counterfactual_accuracy, causal_sensitivity_abs
from src.utils.common_paths import DATASETS_PATH
from src.utils.meta_run import save_experiment
from src.utils.pickle import read_configs


def run_atomic_query(dm, exp_config, probe):
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

    # get intervention from config
    intervention = exp_config['intervention_manager'].from_config(exp_config)

    x, x_cf = dm.get_activations_counterfactual_pairs(_file_names, _configs, intervention)

    # probe default and counterfactual predictions
    preds = probe.predict_proba(x) #[:, 1] # (probe.predict(x) > 0.5).astype(int)
    preds_cf = probe.predict_proba(x_cf) #[:, 1]#(probe.predict(x_cf) > 0.5).astype
    # distr = preds - preds_cf

    # np.set_printoptions(suppress=True, precision=3)
    # print(preds[:10])
    # print(preds_cf[:10])
    # print("preds", preds)
    # print("preds_cf", preds_cf)

    y_gt = np.ones(len(preds))
    y_gt_i = np.zeros(len(preds_cf))

    # metrics
    sensitiviy = causal_sensitivity(preds, preds_cf, y_gt)
    sensitiviy_filtered = causal_sensitivity(preds, preds_cf, y_gt, filtered=True)
    sensitiviy_abs = causal_sensitivity_abs(preds, preds_cf, y_gt)
    sensitiviy_abs_filtered = causal_sensitivity_abs(preds, preds_cf, y_gt, filtered=True)

    cf_accuracy = round(counterfactual_accuracy(preds, preds_cf, y_gt, y_gt_i,), 4)
    # thresholds, coverages, reliabilities, counts = coverage_reliability_curve(preds, preds_cf)

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

    print(f"[INFO] Causal sensitivity (all):      {sensitiviy:.2f}")
    print(f"[INFO] Causal sensitivity (filtered): {sensitiviy_filtered:.2f}")
    print(f"[INFO] Counterfactual accuracy: {cf_accuracy:.2f}")
    del _file_names, _configs, train_configs


def run_atomic_query_new(dm, exp_config, probe):
    # prepare necessary data for intervention
    train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
                                           datasets_path=str(DATASETS_PATH),
                                           as_dict=True)

    # "invariance"
    if exp_config['out_concept'] != exp_config['protected_concept']:
        _file_names = dm.get_filenames_condition_test(exp_config['out_concept'], exp_config['protected_concept'], target_value=None)
        if _file_names.empty:
            print(f"Impossible combination of concepts: Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}")
            save_path = save_experiment(exp_config)
            print(f"[INFO] Saved to {save_path}")
            return

    # "causal sensitivity"
    else: # exp_config['out_concept'] == exp_config['protected']
        _file_names = dm.get_filenames(dm.test, only_positive=True)

    _configs = filter_configs(train_configs, _file_names)

    # get intervention from config
    intervention = exp_config['intervention_manager'].from_config(exp_config)

    x, x_cf = dm.get_activations_counterfactual_pairs(_file_names, _configs, intervention)

    # probe default and counterfactual predictions
    preds = probe.predict_proba(x) #[:, 1] # (probe.predict(x) > 0.5).astype(int)
    preds_cf = probe.predict_proba(x_cf) #[:, 1]#(probe.predict(x_cf) > 0.5).astype
    # distr = preds - preds_cf

    # np.set_printoptions(suppress=True, precision=3)
    # print(preds[:10])
    # print(preds_cf[:10])
    # print("preds", preds)
    # print("preds_cf", preds_cf)

    y_gt = np.ones(len(preds))
    y_gt_i = np.zeros(len(preds_cf))

    # metrics
    sensitiviy = causal_sensitivity(preds, preds_cf, y_gt)
    sensitiviy_filtered = causal_sensitivity(preds, preds_cf, y_gt, filtered=True)
    sensitiviy_abs = causal_sensitivity_abs(preds, preds_cf, y_gt)
    sensitiviy_abs_filtered = causal_sensitivity_abs(preds, preds_cf, y_gt, filtered=True)

    cf_accuracy = round(counterfactual_accuracy(preds, preds_cf, y_gt, y_gt_i,), 4)
    # thresholds, coverages, reliabilities, counts = coverage_reliability_curve(preds, preds_cf)

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

    print(f"[INFO] Causal sensitivity (all):      {sensitiviy:.2f}")
    print(f"[INFO] Causal sensitivity (filtered): {sensitiviy_filtered:.2f}")
    print(f"[INFO] Counterfactual accuracy: {cf_accuracy:.2f}")
    del _file_names, _configs, train_configs