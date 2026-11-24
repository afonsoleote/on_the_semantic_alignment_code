from experiments.disaligned_arch.config import get_all_configs
from experiments.disaligned_arch.main import main
from src.utils.meta_run import save_summary, runner

if __name__ == '__main__':
    runs = [0]
    exp_cfgs = get_all_configs(runs=[0,1,2,3,4])
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])