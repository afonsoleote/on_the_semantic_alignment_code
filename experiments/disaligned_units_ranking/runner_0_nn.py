from experiments.disaligned_units_ranking.config import get_all_configs
from experiments.disaligned_units_ranking.main import main
from src.utils.meta_run import save_summary, runner

if __name__ == '__main__':
    runs = [0]
    probes = ["nn_standard"]
    exp_cfgs = get_all_configs(runs=runs, probes=probes)
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])