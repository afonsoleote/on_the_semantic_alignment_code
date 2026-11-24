import os

from configs import ExperimentConfig
from configs.dataset import XTRAINSConfig
from configs.main_model import TypeAConfig
from src.generation.xtrains.intervention_registry import InterventionManagerXTRAINS


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


def get_config(config, dataset_cfg, model_cfg, concept, protected_concept, seed, probe, intervention_mode, render_fn, conditional, disalign, region_feats, out_dir=None):
    config['seed'] = seed
    config['dataset_cfg'] = dataset_cfg
    config['model_cfg'] = model_cfg
    config['out_concept'] = concept
    config['protected_concept'] = protected_concept
    config['probe'] = probe
    config['intervention_mode'] = intervention_mode
    config['render_fn'] = render_fn
    config['conditional'] = conditional
    config['disalign'] = disalign
    config['region_feats'] = region_feats

    cfg = ExperimentConfig(config)


    return cfg


def get_all_configs(runs=None, probes=None, intervention_modes=None, render_fns=None, model_cfgs=None, concepts=None, protected_dict=None, conditionals=None):
    configs = []

    # set experiment independent variables
    if runs is None:
        runs = [0, 1, 2, 3, 4]
    if probes is None:
        probes = ["linear",]
    if intervention_modes is None:
        intervention_modes = ["sample"]
    if render_fns is None:
        render_fns = ["render_in_mem"]



    regimes = {
        "regime_1": {
            "main_model": TypeAConfig,
            "concept": "PassengerCar",
            "protected": ["PassengerCar", "ReinforcedCar"],
            "conditionals": {
                "PassengerCar": [
                    None,
                    [(1, "FreightWagon")],
                    [(0, "FreightWagon")],
                ],
                "ReinforcedCar": [
                    None,
                    [(1, "FreightWagon")],
                    [(0, "FreightWagon")],
                ]
            },
            "disalign": [
                ("corr", 0.0, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.1, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.2, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.4, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.5, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.6, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.7, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.8, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.85, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.9, "PassengerCar", "ReinforcedCar"),
                ("corr", 0.95, "PassengerCar", "ReinforcedCar"),
                ("corr", 1.0, "PassengerCar", "ReinforcedCar"),

            ],
            "region_feats": ["FreightWagon"]
        },
    }

    # generate all configs
    print("[INFO] Generating all configs")
    for dataset_cfg in [XTRAINSConfig]:
        for run_seed in runs:
            for probe in probes:
                for intervention_mode in intervention_modes:
                    for render_fn in render_fns:
                        for regime_key in regimes:
                            regime = regimes[regime_key]
                            for disalign in regime['disalign']:
                                model_cfg = regime["main_model"]
                                concept = regime["concept"]
                                protected_list = regime["protected"]
                                for protected in protected_list:
                                    conditionals = regime["conditionals"][protected]
                                    for conditional in conditionals:
                                        cfg = get_config(get_common_config(),
                                                         dataset_cfg,
                                                         model_cfg,
                                                         concept,
                                                         protected,
                                                         run_seed,
                                                         probe,
                                                         intervention_mode,
                                                         render_fn,
                                                         conditional,
                                                         disalign,
                                                         regime["region_feats"])
                                        probe_path_id = f"{run_seed}/{regime_key}/{model_cfg.id}/{disalign[0]}/{disalign[1]}/{probe}/{concept}"

                                        out_dir = (f"{probe_path_id}/{protected}/{conditional}/"
                                                   f"{intervention_mode}/{render_fn}")

                                        cfg["out_dir"] = out_dir
                                        cfg["probe_path_id"] = probe_path_id

                                        configs.append(cfg)

    print(f"[INFO] Generated {len(configs)} configs.")
    return configs

if __name__ == "__main__":
    cfgs = get_all_configs()
    print(f"[INFO] Generated {len(cfgs)} configs.")
    print(len(cfgs))
    for c in cfgs:
        print(c["out_dir"])
