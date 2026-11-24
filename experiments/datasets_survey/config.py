import os

from configs import ExperimentConfig
from configs.dataset import XTRAINSConfig, GTSRBConfig, CUBConfig, CelebaConfig, AdultConfig

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

    return config

def get_config(config, dataset_cfg, model_cfg, concept, protected_concept, seed, probe, out_dir=None):
    config['seed'] = seed
    config['dataset_cfg'] = dataset_cfg
    config['model_cfg'] = model_cfg
    config['out_concept'] = concept
    config['protected_concept'] = protected_concept
    config['probe'] = probe

    cfg = ExperimentConfig(config)

    return cfg


def get_all_configs(runs=None, dataset_cfgs=None, probes=None):
    configs = []

    if dataset_cfgs is None:
        dataset_cfgs = [GTSRBConfig, CUBConfig, CelebaConfig, AdultConfig] # [XTRAINSConfig, GTSRBConfig, CUBConfig, CelebaConfig, AdultConfig] # [XTRAINSConfig, GTSRBConfig, CUBConfig, CelebaConfig, AdultConfig]
    if runs is None:
        runs = [0, 1, 2, 3, 4] # [0]
    if probes is None:
        probes = ["linear", "lgbm", "ridge", "nn_standard"]

    print("[INFO] Generating all configs")
    for run_seed in runs:
        for dataset_cfg  in dataset_cfgs:
            for main_cfg in dataset_cfg.models:
                for probe in probes:
                    for concept in main_cfg.concepts:
                        protected_list = list(main_cfg.protected)
                        if dataset_cfg.id == "xtrains" and concept not in protected_list:
                                protected_list.append(concept)
                        for protected in protected_list:
                            cfg = get_config(get_common_config(),
                                             dataset_cfg,
                                             main_cfg,
                                             concept,
                                             protected,
                                             run_seed,
                                             probe,)

                            probe_path_id = f"{run_seed}/{main_cfg.id}/{probe}/{concept}"

                            out_dir = f"{probe_path_id}/{protected}"

                            cfg["out_dir"] = out_dir
                            cfg["probe_path_id"] = probe_path_id

                            configs.append(cfg)

    print(f"[INFO] Generated {len(configs)} configs.")
    return configs

if __name__ == "__main__":
    cfgs = get_all_configs()
    print(f"[INFO] Generated {len(cfgs)} configs.")


