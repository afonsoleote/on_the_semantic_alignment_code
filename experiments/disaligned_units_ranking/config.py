import os
import pickle

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
        probes = ["linear", "nn_standard"]
    if intervention_modes is None:
        intervention_modes = ["basic"]
    if render_fns is None:
        render_fns = ["render_in_mem"]

    with open("rankings_full.pickle", "rb") as f:
        rankings = pickle.load(f)

    # keep_rankings = {"linear", "linear_w_3", "linear_w_5", "linear_cumulative"}
    # keep_rankings = {"kde", "kde_w_3", "kde_w_5", "kde_cumulative",
    #                 "combined", "combined_w_3", "combined_w_5", "combined_cumulative",}

    keep_rankings = {
        'replace_sweep_10_full',
        # 'replace_sweep_15_full',
        # 'replace_sweep_20_full',
        # 'replace_sweep_25_full',
        # 'replace_sweep_30_full',
        'interp_sweep_10_full',
        # 'interp_sweep_15_full',
        # 'interp_sweep_20_full',
        # 'interp_sweep_25_full',
        # 'interp_sweep_30_full',
    }

    rankings = {k: v for k, v in rankings.items() if k in keep_rankings}

    # generate all configs
    print("[INFO] Generating all configs")
    for dataset_cfg in [XTRAINSConfig]:
        for run_seed in runs:
            for model_cfg in [TypeAConfig]:
                for concept in ["PassengerCar"]:
                    for protected in ["PassengerCar", "ReinforcedCar"]:
                        for ranking in rankings:
                            print(ranking, len(rankings[ranking]))
                            for i, to_extract in enumerate(rankings[ranking]):
                                for corr_pair in [("PassengerCar", "ReinforcedCar")]:
                                    for corr_level in [0.95]:
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
                                                                     corr_level,)

                                                    probe_path_id = f"{run_seed}/{model_cfg.id}/{probe}/{concept}/{ranking}/{i+1}/{corr_level}"

                                                    out_dir = (f"{probe_path_id}/{protected}/"
                                                               f"{intervention_mode}/{render_fn}")

                                                    cfg["out_dir"] = out_dir
                                                    cfg["probe_path_id"] = probe_path_id
                                                    cfg["ranking"] = i
                                                    configs.append(cfg)

    print(f"[INFO] Generated {len(configs)} configs.")
    return configs

if __name__ == "__main__":
    cfgs = get_all_configs()
    print(f"[INFO] Generated {len(cfgs)} configs.")

    print(len(cfgs))

    for c in cfgs:
        print(c["out_dir"])
