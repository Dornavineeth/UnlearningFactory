import hydra
from omegaconf import DictConfig

def get_config(config_type, name) -> DictConfig:
    hydra.initialize(config_path=config_type)
    cfg = hydra.compose(config_name=f"{name}.yaml")
    return cfg