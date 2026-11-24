import time

import matplotlib.pyplot as plt

from experiments.datasets_survey.config import get_all_configs
from src.data.datamodules_generic import XTRAINSDataModule, GTSRBDataModule, CUBDataModule, \
    CelebaDataModule, AdultDataModule
from src.reliability.metrics import alignment, alignment_abs
from src.utils.meta_run import load_or_train_probe, save_experiment, runner, save_summary

DATAMODULE_BY_ID = {
    'xtrains': XTRAINSDataModule,
    'gtsrb':   GTSRBDataModule,
    'cub':     CUBDataModule,
    'celeba':  CelebaDataModule,
    'adult':   AdultDataModule,
}

def set_data(exp_config):
    dataset_cfg = exp_config['dataset_cfg']
    dm_class = DATAMODULE_BY_ID[dataset_cfg.id]
    dm = dm_class(
        dataset_cfg,
        [exp_config['out_concept']],
        exp_config['batch_size'],
        exp_config['seed'],
        exp_config['device'],
        exp_config['model_cfg'],
    )

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

    names, x, x_cf = dm.get_activations_counterfactual_pairs(exp_config["protected_concept"])

    preds = probe.predict_proba(x)
    preds_cf = probe.predict_proba(x_cf)

    sensitivity = alignment(preds, preds_cf)
    sensitivity_abs = alignment_abs(preds, preds_cf)

    exp_config['sensitivity'] = sensitivity
    exp_config['sensitivity_abs'] = sensitivity_abs

    exp_config['independence'] = 1 - abs(sensitivity)
    exp_config['independence_abs'] = 1 - abs(sensitivity_abs)

    print(f"[INFO] Causal sensitivity (normal):      {sensitivity:.2f}")
    print(f"[INFO] Causal sensitivity (abs):         {sensitivity_abs:.2f}")

    exp_config["time"] = round(time.time() - ts, 3)

    save_path = save_experiment(exp_config)
    print(f"[INFO] Saved to {save_path}")
    print(f"[INFO] Experiment {exp_config["id"]} time: {exp_config["time"]} s")

def debug(exp_config):
    ts = time.time()
    print(f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)
    # probe, train_results = load_or_train_probe(dm, exp_config)
    # exp_config.update(train_results)

    names, x, x_cf = dm.get_counterfactual_pairs(exp_config["protected_concept"])

    print("Amount of samples:",  len(x))
    for i in range(min(5, len(x))):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(x[i].permute(1, 2, 0))
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(x_cf[i].permute(1, 2, 0))
        ax[1].set_title("CF")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()

    exp_config["time"] = round(time.time() - ts, 3)


if __name__ == '__main__':
    exp_cfgs = get_all_configs()
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])