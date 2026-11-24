import pandas as pd
import pickle

from collections import defaultdict
from pathlib import Path

from src.probing.probes.training import train_model
from src.utils.common_paths import OUTPUT_PATH


def group_configs(configs, keys):
    if isinstance(keys, str):
        keys = (keys,)

    groups = defaultdict(list)

    for cfg in configs:
        key = tuple(cfg[k] for k in keys)

        def freeze(x):
            if isinstance(x, dict):
                return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                                    for k, v in x.items()))
            return x

        key = tuple(freeze(cfg[k]) for k in keys)

        if len(key) == 1:
            key = key[0]
        groups[key].append(cfg)

    return groups



def save_experiment(exp_config, output_root="../../output"):
    exp_id = exp_config["id"]

    run_path = Path(output_root) / exp_id / "runs" / exp_config["out_dir"]

    run_path.mkdir(parents=True, exist_ok=True)

    with open(run_path / "results.pickle", "wb") as f:
        pickle.dump(dict(exp_config.items()), f)

    return run_path


def collect_results(exp_id, output_root="../../output"):
    base_path = Path(output_root) / exp_id / "runs"

    rows = []

    for result_file in base_path.rglob("results.pickle"):
        with open(result_file, "rb") as f:
            rows.append(pickle.load(f))

    return pd.DataFrame(rows)

def save_summary(exp_id, output_root="../../output"):
    df = collect_results(exp_id, output_root)

    summary_path = Path(output_root) / exp_id / "summary"
    summary_path.mkdir(parents=True, exist_ok=True)

    csv_path = summary_path / "results.csv"
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Summary saved to {csv_path}")


def load_or_train_probe(dm, exp_config):
    probe_dir = exp_config['probe_dir']
    probe_pickle_path = Path(probe_dir) / 'probe_model.pickle' if probe_dir else None


    if probe_pickle_path and probe_pickle_path.exists():
        print("[INFO] Existing probe. Loading...")
        with open(probe_pickle_path, 'rb') as f:
            probe, train_results = pickle.load(f)
    else:
        print("[INFO] Non-existing probe. Training...")
        probe, train_results = train_model(dm, exp_config)

        if probe_pickle_path:
            probe_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with open(probe_pickle_path, 'wb') as f:
                pickle.dump((probe, train_results), f)

    return probe, train_results

def runner(exp_configs, main):
    total_confs = len(exp_configs)
    exp_id = exp_configs[0]["id"]

    for i, exp_cfg in enumerate(exp_configs):
        gid = exp_cfg["out_dir"]

        run_path = OUTPUT_PATH / exp_id / "runs" / gid
        result_file = run_path / "results.pickle"

        exp_cfg['probe_dir'] = OUTPUT_PATH / exp_id / "runs" / exp_cfg['probe_path_id']

        if result_file.exists():
            print(f"[INFO] Skipping {gid} ({i}/{total_confs})")
            continue

        print(f"[INFO] Starting run {gid} ({i}/{total_confs}) ({(i*100)/total_confs:.3}%)")
        main(exp_cfg)
