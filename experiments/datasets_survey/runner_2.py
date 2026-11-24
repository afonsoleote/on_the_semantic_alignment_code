from experiments.datasets_survey.config import get_all_configs
from experiments.datasets_survey.main import main
from src.utils.meta_run import runner, save_summary

if __name__ == '__main__':
    runs = [2]
    exp_cfgs = get_all_configs(runs=runs)
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])