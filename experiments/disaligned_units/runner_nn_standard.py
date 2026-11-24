from experiments.disaligned_units.config import get_all_configs
from experiments.disaligned_units.main import main
from src.utils.meta_run import save_summary, runner

if __name__ == '__main__':
    probes = ["nn_standard"]
    exp_cfgs = get_all_configs(probes=probes)
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])