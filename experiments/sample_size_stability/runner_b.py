from configs.main_model import TypeBConfig
from experiments.sample_size_stability.config import get_all_configs
from experiments.sample_size_stability.main import main
from src.utils.meta_run import save_summary, runner

if __name__ == '__main__':
    model_cfgs = [TypeBConfig]
    exp_cfgs = get_all_configs(model_cfgs=model_cfgs)
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])