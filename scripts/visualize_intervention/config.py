import os

from configs import DATASETS, ExperimentConfig
from configs.main_model import TypeAConfig
from src.generation.xtrains.intervention_registry import InterventionManagerXTRAINS
from src.utils.meta_run import group_configs


def name():
    return os.path.split(os.path.dirname(__file__))[-1]

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


def get_config(config, dataset_cfg, model_cfg, concept, protected_concept, seed, probe, intervention_mode, render_fn, out_dir=None):
    config['seed'] = seed
    config['dataset_cfg'] = dataset_cfg
    config['model_cfg'] = model_cfg
    config['out_concept'] = concept
    config['protected_concept'] = protected_concept
    config['probe'] = probe
    config['intervention_mode'] = intervention_mode
    config['render_fn'] = render_fn

    cfg = ExperimentConfig(config)


    return cfg


def get_all_configs(runs=None, probes=None, intervention_modes=None, render_fns=None, model_cfgs=None, concepts=None, protected_list=None):
    configs = []

    # set experiment independent variables
    if runs is None:
        runs = [1] # [0, 1, 2, 3, 4]
    if probes is None:
        probes = ["linear",] #"lgbm"] #, "lgbm", "ridge", "nn"] # ["linear", "lgbm", ]
    if intervention_modes is None:
        intervention_modes = ["sample"] # ["basic", "sample"]
    if render_fns is None:
        render_fns = ["render_in_mem"] #, "render_in_mem_random"] # ["render_in_mem",  "render_in_mem_random"]
    if model_cfgs is None:
        model_cfgs = [TypeAConfig]
    if concepts is None:
        concepts = ["ReinforcedCar"]
    if protected_list is None:
        protected_list = ["ReinforcedCar"]

    # generate all configs
    print("[INFO] Generating all configs")
    for dataset_cfg in DATASETS:
        for run_seed in runs:
            for model_cfg in model_cfgs:
                for probe in probes:
                    for concept in concepts:

                        _protected_list = protected_list.copy()

                        if concept not in _protected_list:
                            _protected_list.insert(0, concept)

                        for protected in _protected_list:
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
                                                     render_fn,)

                                    probe_path_id = f"{run_seed}/{model_cfg.id}/{probe}/{concept}"

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

    dict_cfgs = group_configs(cfgs, ('main_name', 'render_fn'))
    print(len(dict_cfgs))
    for k in dict_cfgs:
        print(len(dict_cfgs[k]))
