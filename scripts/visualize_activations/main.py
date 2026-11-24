import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from scripts.visualize_intervention.config import get_all_configs
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.generation.xtrains.utils import filter_configs
from src.utils.common_paths import DATASETS_PATH
from src.utils.pickle import read_configs

BLUE = "#4c72b0"
ORANGE = "#dd8452"
GRAY = "0.6"


def project(x, x_cf, n_components):
    x = np.asarray(x).reshape(len(x), -1)
    x_cf = np.asarray(x_cf).reshape(len(x_cf), -1)

    pca = PCA(n_components=n_components)
    pca.fit(np.vstack([x, x_cf]))

    return pca.transform(x), pca.transform(x_cf)


def plot_2d(x, x_cf, connect=False):
    p, p_cf = project(x, x_cf, 2)

    fig, ax = plt.subplots(figsize=(5, 4))

    if connect:
        for i in range(len(p)):
            ax.plot([p[i, 0], p_cf[i, 0]],
                    [p[i, 1], p_cf[i, 1]],
                    ls="--", lw=0.5, color=GRAY)

    ax.scatter(p[:, 0], p[:, 1], s=10, color=BLUE, label="x")
    ax.scatter(p_cf[:, 0], p_cf[:, 1], s=10, color=ORANGE, label="x_cf")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.show()


def plot_3d(x, x_cf, connect=False):
    p, p_cf = project(x, x_cf, 3)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")

    if connect:
        for i in range(len(p)):
            ax.plot([p[i, 0], p_cf[i, 0]],
                    [p[i, 1], p_cf[i, 1]],
                    [p[i, 2], p_cf[i, 2]],
                    ls="--", lw=0.5, color=GRAY)

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=10, color=BLUE, label="x")
    ax.scatter(p_cf[:, 0], p_cf[:, 1], p_cf[:, 2], s=10, color=ORANGE, label="x_cf")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(frameon=False)
    plt.show()


def plot_neuron_change(x, x_cf):
    x = np.asarray(x).reshape(len(x), -1)
    x_cf = np.asarray(x_cf).reshape(len(x_cf), -1)

    diff = x - x_cf
    change = np.abs(diff).mean(axis=0)
    neurons = np.arange(x.shape[1])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(neurons, change, width=1.0, color=BLUE, linewidth=0)

    ax.set_xlabel("Neuron")
    ax.set_ylabel(r"Mean $|x - x_{cf}|$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

def set_data(exp_config):
    dm = ProbingDataModuleXTRAINS([exp_config['out_concept']],
                           'image_id',
                                  exp_config['batch_size'],
                                  exp_config['dataset_cfg'].labels_path,
                                  exp_config['dataset_cfg'].images_path,
                                  exp_config['device'],
                                  exp_config['seed'],
                                  exp_config['model_cfg'].path,
                                  # exp_config['model_cfg'].layers
                                  {"36": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
                                  )

    dm.set_default_splits(exp_config['tr_size'],
                          exp_config['val_size'],
                          exp_config['test_size'])
    return dm

def main(exp_config):
    print(
        f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)
    train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
                                           datasets_path=str(DATASETS_PATH),
                                           as_dict=True)

    # "invariance"
    if exp_config['out_concept'] != exp_config['protected_concept']:
        _file_names = dm.get_filenames_condition(exp_config['out_concept'], exp_config['protected_concept'])

    # "causal sensitivity"
    else:
        _file_names = dm.get_filenames(dm.test, only_positive=True)

    _configs = filter_configs(train_configs, _file_names)

    intervention = exp_config['intervention_manager'].from_config(exp_config)

    x, x_cf = dm.get_activations_counterfactual_pairs(_file_names, _configs, intervention)

    plot_2d(x, x_cf)
    plot_2d(x, x_cf, connect=True)
    plot_3d(x, x_cf)
    plot_3d(x, x_cf, connect=True)
    plot_neuron_change(x, x_cf)

    del _file_names, _configs, train_configs


if __name__ == '__main__':
    exp_configs = get_all_configs()

    for exp_cfg in exp_configs:
        main(exp_cfg)



