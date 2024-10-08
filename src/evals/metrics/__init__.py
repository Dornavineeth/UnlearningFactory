from omegaconf import DictConfig

from .memorization import (
    qa_prob,
    qa_paraphrased_prob,
    qa_perturbed_prob,
    qa_rouge,
    qa_paraphrased_rouge,
    qa_perturbed_rouge
)


def _get_single_metric(metric_name, metric_cfg, **kwargs):
    if metric_name == 'TOFU_QA_Prob':
        return qa_prob
    elif metric_name == 'TOFU_QA_P_Prob':
        return qa_paraphrased_prob
    elif metric_name == 'TOFU_QA_PT_Prob':
        return qa_perturbed_prob
    elif metric_name == 'TOFU_QA_ROUGE':
        return qa_rouge
    elif metric_name == "TOFU_QA_P_ROUGE":
        return qa_paraphrased_rouge
    elif metric_name == "TOFU_QA_PT_ROUGE":
        return qa_perturbed_rouge
    else:
        raise NotImplementedError(f"{metric_name} not implemented")

def get_metrics(metric_cfgs: DictConfig, **kwargs):
    metrics = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metrics[metric_name] = _get_single_metric(metric_name, metric_cfg, **kwargs)
    
    return metrics