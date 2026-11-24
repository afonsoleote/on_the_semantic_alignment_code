import time

from experiments.architectures.config import get_all_configs
from experiments.utils import run_atomic_query
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.utils.meta_run import save_experiment, save_summary, load_or_train_probe, runner


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
    print(f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)
    probe, train_results = load_or_train_probe(dm, exp_config)
    exp_config.update(train_results)

    run_atomic_query(dm, exp_config, probe)

    exp_config["time"] = round(time.time() - ts, 3)

    save_path = save_experiment(exp_config)
    print(f"[INFO] Saved to {save_path}")
    print(f"[INFO] Experiment {exp_config["id"]} time: {exp_config["time"]} s")



if __name__ == '__main__':
    exp_cfgs = get_all_configs()
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])



