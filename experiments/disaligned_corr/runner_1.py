from experiments.disaligned_corr.config import get_all_configs
from experiments.disaligned_corr.main import main
from src.utils.meta_run import save_summary, runner

if __name__ == '__main__':
    runs = [1, 2]
    exp_cfgs = get_all_configs(runs=runs)
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])