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


def get_config(config, dataset_cfg, model_cfg, concept, protected_concept, seed, probe, intervention_mode, render_fn, corr_pair, corr_level, epochs, out_dir=None):
    config['seed'] = seed
    config['dataset_cfg'] = dataset_cfg
    config['model_cfg'] = model_cfg
    config['out_concept'] = concept
    config['protected_concept'] = protected_concept
    config['probe'] = probe
    config['intervention_mode'] = intervention_mode
    config['render_fn'] = render_fn
    config['corr_pair'] = corr_pair
    config['corr'] = corr_level
    config['epochs'] = epochs

    cfg = ExperimentConfig(config)


    return cfg


def get_all_configs(runs=None, probes=None, intervention_modes=None, render_fns=None, corr_levels=None, concepts=None, epochs=None):
    configs = []

    # set experiment independent variables
    if runs is None:
        runs = [0, 1, 2, 3, 4,]
    if probes is None:
        probes = ["nn_linear", "nn_standard", "nn_medium", "nn_large",]
    if intervention_modes is None:
        intervention_modes = ["basic"]
    if render_fns is None:
        render_fns = ["render_in_mem"]
        corr_levels = [0.95,]
    if epochs is None:
        epochs = [150,]
    if concepts is None:
        concepts = ["PassengerCar",]


    concept_corr_pair_dict = {
        "PassengerCar": ("PassengerCar", "ReinforcedCar"),
        "WarTrain": ("PassengerCar", "FreightWagon"),
        "OpenRoofCar": ("OpenRoofCar", "PassengerCar"),
    }

    # generate all configs
    print("[INFO] Generating all configs")
    for dataset_cfg in [XTRAINSConfig]:
        for run_seed in runs:
            for model_cfg in [TypeAConfig]:
                for probe in probes:
                    for epoch in epochs:
                        for concept in concepts:
                            corr_pair = concept_corr_pair_dict[concept]
                            for protected in corr_pair:
                                for corr_level in corr_levels:
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
                                                             corr_pair,
                                                             corr_level,
                                                             epoch,
                                                             )

                                            probe_path_id = f"{run_seed}/{model_cfg.id}/{probe}/{concept}/{corr_level}/{epoch}"

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
