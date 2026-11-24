from pathlib import Path

from matplotlib import pyplot as plt

from scripts.visualize_intervention.config import get_all_configs
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.generation.xtrains.utils import filter_configs
from src.probing.probes.training import train_model
from src.utils.common_paths import DATASETS_PATH
from src.utils.pickle import read_configs
from src.utils.plots.general_purpose import show_image_tensor_pairs_with_score, show_image_tensor_pairs

def show_image_tensor_pairs_save(x, x_cf, n=5, configs=None, save_dir="saved_images"):
    save_dir = Path(save_dir)   
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(n, len(x))):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        print(i)
        if configs is not None:
            print(configs[i]["id"])

        ax[0].imshow(x[i].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(x_cf[i].permute(1, 2, 0).cpu().numpy())
        ax[1].set_title("Counterfactual")
        ax[1].axis("off")

        def on_key(event):
            if event.key == "s":
                image_id = configs[i]["id"] if configs is not None else i

                fig_orig = save_dir / f"{image_id}_original.jpg"
                fig_cf = save_dir / f"{image_id}_counterfactual.jpg"

                plt.imsave(
                    fig_orig,
                    x[i].permute(1, 2, 0).cpu().numpy(),
                    pil_kwargs={"quality": 100}
                )

                plt.imsave(
                    fig_cf,
                    x_cf[i].permute(1, 2, 0).cpu().numpy(),
                    pil_kwargs={"quality": 100}
                )

                print(f"Saved: {fig_orig}")
                print(f"Saved: {fig_cf}")

            elif event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.tight_layout()
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
                                  exp_config['model_cfg'].layers)

    dm.set_default_splits(exp_config['tr_size'],
                          exp_config['val_size'],
                          exp_config['test_size'])
    return dm

def main(exp_config):
    print(
        f"[INFO] Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}, Probe: {exp_config["probe"]}")

    dm = set_data(exp_config)

    # no need to train a probe here

    # probe, train_results = train_model(dm, exp_config)
    # exp_config.update(train_results)

    # prepare necessary data for intervention
    train_configs = read_configs(exp_config['dataset_cfg'].configs_path,
                                           datasets_path=str(DATASETS_PATH),
                                           as_dict=True)

    # "invariance"
    if exp_config['out_concept'] != exp_config['protected_concept']:
        _file_names = dm.get_filenames_condition(exp_config['out_concept'], exp_config['protected_concept'])
        if _file_names.empty:
            print(f"Impossible combination of concepts: Concept: {exp_config["out_concept"]}, Protected: {exp_config["protected_concept"]}")
            # save_path = save_experiment(exp_config)
            # print(f"[INFO] Saved to {save_path}")
            return

    # "causal sensitivity"
    else: # exp_config['out_concept'] == exp_config['protected']
        _file_names = dm.get_filenames(dm.test, only_positive=True)

    _configs = filter_configs(train_configs, _file_names)

    # get intervention from config
    intervention = exp_config['intervention_manager'].from_config(exp_config)

    x_img, x_img_cf = dm.get_counterfactual_pairs(_file_names, _configs, intervention)

    show_image_tensor_pairs_save(x_img, x_img_cf, 100, save_dir="saved_images_jpeg")


if __name__ == '__main__':
    exp_configs = get_all_configs()

    for exp_cfg in exp_configs:
        main(exp_cfg)



