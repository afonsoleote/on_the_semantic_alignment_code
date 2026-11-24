import os

from configs import DATASETS, ExperimentConfig
from configs.dataset import XTRAINSConfig
from configs.main_model import TypeAConfig
from src.generation.xtrains.intervention_registry import InterventionManagerXTRAINS
from src.utils.meta_run import group_configs


def name():
    return os.path.split(os.path.dirname(__file__))[-1]

# def identity_keys():
#     return

def get_common_config():
    config = dict()

    config['id'] = name()

    config['device'] = 'cpu'
    config['batch_size'] = 32

    config['tr_size'] = 800
    config['val_size'] = 200
    config['test_size'] = 1000

    config['intervention_manager'] = InterventionManagerXTRAINS()

    return config


def get_config(config, dataset_cfg, model_cfg, concept, protected_concept, seed, probe, intervention_mode, render_fn, to_extract, corr_pair, corr_level, out_dir=None):
    config['seed'] = seed
    config['dataset_cfg'] = dataset_cfg
    config['model_cfg'] = model_cfg
    config['out_concept'] = concept
    config['protected_concept'] = protected_concept
    config['probe'] = probe
    config['intervention_mode'] = intervention_mode
    config['render_fn'] = render_fn
    config['to_extract'] = to_extract
    config['corr_pair'] = corr_pair
    config['corr'] = corr_level

    cfg = ExperimentConfig(config)


    return cfg


def get_all_configs(runs=None, probes=None, intervention_modes=None, render_fns=None):
    configs = []

    # set experiment independent variables
    if runs is None:
        runs = [0, 1, 2, 3, 4]
    if probes is None:
        probes = ["linear", "nn_standard",] #"lgbm"] #, "lgbm", "ridge", "nn"] # ["linear", "lgbm", ]
    if intervention_modes is None:
        intervention_modes = ["basic"] # ["basic", "sample"]
    if render_fns is None:
        render_fns = ["render_in_mem"] #, "render_in_mem_random"] # ["render_in_mem",  "render_in_mem_random"]


    # generate all configs
    print("[INFO] Generating all configs")
    for dataset_cfg in [XTRAINSConfig]:
        for run_seed in runs:
            for model_cfg in [TypeAConfig]:
                for concept in ["PassengerCar"]:
                    for protected in ["PassengerCar", "ReinforcedCar"]:
                        for to_extract in [{"36": [4]}, {"36": [12]}, {"36": [4, 12]}]:
                            for corr_pair in [("PassengerCar", "ReinforcedCar")]:
                                for corr_level in [0.9, 0.95, 0.98, 0.99]: # [0.8,0.9,0.95,0.98,0.99]:
                                    for probe in probes:
                                        for intervention_mode in intervention_modes:
                                            for render_fn in render_fns:
                                                # yield
                                                cfg = get_config(get_common_config(),
                                                                 dataset_cfg,
                                                                 model_cfg,
                                                                 concept,
                                                                 protected,
                                                                 run_seed,
                                                                 probe,
                                                                 intervention_mode,
                                                                 render_fn,
                                                                 to_extract,
                                                                 corr_pair,
                                                                 corr_level)

                                                probe_path_id = f"{run_seed}/{model_cfg.id}/{probe}/{concept}/{to_extract["36"]}/{corr_level}/"

                                                out_dir = (f"{probe_path_id}/{protected}/"
                                                           f"{intervention_mode}/{render_fn}")

                                                cfg["out_dir"] = out_dir
                                                cfg["probe_path_id"] = probe_path_id

                                                configs.append(cfg)

    print(f"[INFO] Generated {len(configs)} configs.")
    return configs

if __name__ == "__main__":
    cfgs = get_all_configs()
    print(f"[INFO] Generated {len(cfgs)} configs.")

    for c in cfgs:
        print(c['out_dir'])
    dict_cfgs = group_configs(cfgs, ("out_concept", "to_extract", "corr"))
    print(len(dict_cfgs))
    for k in dict_cfgs:
        print(len(dict_cfgs[k]))
