from .tofu import TOFU_QA

# TODO add tofu wiki support
def get_dataset(name, dataset_config):
    if name == "TOFU_QA":
        return TOFU_QA(**dataset_config)
    else:
        raise NotImplementedError