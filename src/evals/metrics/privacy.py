import numpy as np
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals, eval_minKpc_neg_logprob


@unlearning_metric(name="forget_quality")
def forget_quality(model, **kwargs):
    forget_tr_stats = np.array(
        [
            evals["score"]
            for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
        ]
    )
    retain_tr_stats = np.array(
        [
            evals["score"]
            for evals in kwargs["reference_logs"]["retain_model_logs"]["retain"][
                "value_by_index"
            ].values()
        ]
    )
    fq = ks_2samp(forget_tr_stats, retain_tr_stats)
    return {"agg_value": fq.pvalue}


@unlearning_metric(name="minKpc_negative_logprob")
def minKpc_negative_logprob(model, **kwargs):
    """Compute the min-k percentile average of token-wise model probabilities by data points"""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"percentile": kwargs["percentile_K"]}
    return {
        "value_by_index": run_batchwise_evals(
            model,
            dataloader,
            eval_minKpc_neg_logprob,
            fun_args,
            "Calculating avg token-wise lowest K% percentile logprobs across batches",
        )
    }


@unlearning_metric(name="relative_auc")
def relative_auc(model, **kwargs):
    """Compute the auc score of an MIA attack wrt model scores on a victim and holdout set"""

    def sweep(ppl, y):
        fpr, tpr, _ = get_roc_curve(y, -ppl)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        return fpr, tpr, get_auc(fpr, tpr), acc

    forget_scores = kwargs["pre_compute"]["forget"]["value_by_index"].values()
    forget_scores = [elem["score"] for elem in forget_scores]
    forget_holdout_scores = kwargs["pre_compute"]["holdout"]["value_by_index"].values()
    forget_holdout_scores = [elem["score"] for elem in forget_holdout_scores]
    scores = np.array(forget_scores + forget_holdout_scores)
    # in MUSE the scores are -mean(min k% log-probs) for some reason so flip the 1 and 0
    labels = np.array([0] * len(forget_scores) + [1] * len(forget_holdout_scores))

    _, _, auc_score, acc = sweep(scores, labels)

    output = {
        "acc": acc,
        "auc": auc_score,
    }
    retain_auc_score = kwargs["ref_value"]

    reference_logs = kwargs.get("reference_logs", None)
    if reference_logs:
        retain_scores = reference_logs["retain_model_logs"]["retain"][
            "value_by_index"
        ].values()
        retain_scores = [elem["score"] for elem in retain_scores]
        retain_holdout_scores = reference_logs["retain_model_logs"]["holdout"][
            "value_by_index"
        ].values()
        retain_holdout_scores = [elem["score"] for elem in retain_holdout_scores]
        scores = np.array(retain_scores + retain_holdout_scores)
        labels = np.array([0] * len(retain_scores) + [1] * len(retain_holdout_scores))
        _, _, retain_auc_score, retain_acc = sweep(scores, labels)
        output.update({"retain_acc": retain_acc, "retain_auc_score": retain_auc_score})

    output.update(
        {
            "agg_value": (auc_score - retain_auc_score)
            / (retain_auc_score)
            * 100  # privleak score in muse
        }
    )
    return output
