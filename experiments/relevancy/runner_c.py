from configs.main_model import TypeCConfig
from experiments.relevancy.config import get_all_configs
from experiments.relevancy.main import main
from src.utils.meta_run import save_summary, runner

if __name__ == '__main__':
    model_cfgs = [TypeCConfig]
    exp_cfgs = get_all_configs(model_cfgs=model_cfgs)
    runner(exp_cfgs, main)
    save_summary(exp_cfgs[0]["id"])