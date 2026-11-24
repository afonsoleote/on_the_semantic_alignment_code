import os

from configs import DATASETS, ExperimentConfig
from configs.dataset import XTRAINSConfig
from configs.main_model import TypeAConfig
from src.generation.xtrains.intervention_registry import InterventionManagerXTRAINS


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


def get_config(config, dataset_cfg, model_cfg, concepts, seed, probe, intervention_mode, render_fn, to_extract, out_dir=None):
    config['seed'] = seed
    config['dataset_cfg'] = dataset_cfg
    config['model_cfg'] = model_cfg
    config['out_concept'] = concepts
    config['probe'] = probe
    config['intervention_mode'] = intervention_mode
    config['render_fn'] = render_fn
    config['to_extract'] = to_extract

    cfg = ExperimentConfig(config)


    return cfg


def get_all_configs(runs=None, probes=None, intervention_modes=None, render_fns=None, model_cfgs=None, concepts_list=None, protected_list=None):
    configs = []

    # set experiment independent variables
    if runs is None:
        runs = [0] # [0, 1, 2, 3, 4]
    if probes is None:
        probes = ["linear",]
    if intervention_modes is None:
        intervention_modes = ["sample"]
    if render_fns is None:
        render_fns = ["render_in_mem"]
    if model_cfgs is None:
        model_cfgs = [TypeAConfig]
    if concepts_list is None:
        concepts_list = ["PassengerCar", "ReinforcedCar"]

    layer_neuron_dict = {
        30: [i for i in range(64)],
        33: [i for i in range(32)],
        36: [i for i in range(16)]

    }
    print(layer_neuron_dict)

    # generate all configs
    print("[INFO] Generating all configs")
    for dataset_cfg in [XTRAINSConfig]:
        for run_seed in runs:
            for model_cfg in model_cfgs:
                for probe in probes:
                    for concept in concepts_list:
                        for layer in layer_neuron_dict:
                            #for neuron in range(0,16):
                            for neuron in layer_neuron_dict[layer]:
                                to_extract = {f"{layer}":[neuron]}
                                for intervention_mode in intervention_modes:
                                    for render_fn in render_fns:
                                        # yield
                                        cfg = get_config(get_common_config(),
                                                         dataset_cfg,
                                                         model_cfg,
                                                         concept,
                                                         run_seed,
                                                         probe,
                                                         intervention_mode,
                                                         render_fn,
                                                         to_extract)

                                        probe_path_id = f"{run_seed}/{model_cfg.id}/{probe}/{concept}"

                                        out_dir = (f"{probe_path_id}/{to_extract}/"
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


