import time
from pathlib import Path

from experiments.relevancy.config import get_all_configs
from experiments.utils import run_atomic_query
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.generation.xtrains.utils import filter_configs
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
    print(
        f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)

    probe, train_results = load_or_train_probe(dm, exp_config)
    exp_config.update(train_results)

    run_atomic_query(dm, exp_config, probe)

    exp_config["time"] = round(time.time() - ts, 3)

    save_path = save_experiment(exp_config)
    print(f"[INFO] Saved to {save_path}")
    print(f"[INFO] Experiment {exp_config["id"]} time: {exp_config["time"]} s")

# for cf probes


# def main(exp_config):
#     ts = time.time()
#     print(
#         f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")
#
#     dm = set_data(exp_config)
#
#     probe_dir = exp_config['probe_dir']
#     probe_pickle_path = Path(probe_dir) / 'probe_model.pickle' if probe_dir else None
#
#     if not (probe_pickle_path and probe_pickle_path.exists()):
#         if exp_config['probe'] == "linear_cf":
#             train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
#                                          datasets_path=str(DATASETS_PATH),
#                                          as_dict=True)
#             _file_names = dm.get_filenames(dm.tr, only_positive=True)
#             _configs = filter_configs(train_configs, _file_names)
#             intervention = exp_config['intervention_manager'].from_config_self_intervention(exp_config)
#             x, x_cf = dm.get_activations_counterfactual_pairs(_file_names, _configs, intervention)
#             # dm.set_cf_train(x, x_cf)
#
#     probe, train_results = load_or_train_probe(dm, exp_config)
#     exp_config.update(train_results)
#
#     run_atomic_query(dm, exp_config, probe)
#
#     exp_config["time"] = round(time.time() - ts, 3)
#
#     save_path = save_experiment(exp_config)
#     print(f"[INFO] Saved to {save_path}")
#     print(f"[INFO] Experiment {exp_config["id"]} time: {exp_config["time"]} s")

if __name__ == '__main__':
    exp_cfgs = get_all_configs()
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])



