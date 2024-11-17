import logging
import numpy as np
import scipy as sc
from torch.utils.data import DataLoader

from evals.metrics.utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    run_batchwise_evals,
)
from evals.metrics.base import unlearning_metric

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)


@unlearning_metric(name="probability")
def probability(model, **kwargs):
    # returns the prob and avg_loss in scores
    # aggregate the prob values
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    prob_values = np.array([evals["prob"] for evals in scores_by_index.values()])
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


@unlearning_metric(name="probability_w_options")
def probability_w_options(model, **kwargs):
    """Normalize probabilities of correct answers against false answers for
    open-ended datasets, returning the aggregated value and per-index probabilities."""
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answers_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answers_results.keys())
    assert correct_indices == wrong_indices
    correct = [evals["prob"] for evals in correct_answer_results.values()]
    all_wrong = [evals["prob"] for evals in wrong_answers_results.values()]

    correct = np.array(correct)
    all_wrong = np.array(all_wrong)
    wrong = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))

    probs = correct / (correct + wrong)

    value_by_index = dict(zip(correct_indices, [{"prob": val} for val in probs]))
    return {"agg_value": np.mean(probs), "value_by_index": value_by_index}


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics (rouge1_recall, rougeL_recall, input, ground_truth,
    generation), aggregate the rougeL_recall values, and return the aggregated value
    along with per-index scores."""
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity,
        fun_args,
        "Calculating text similarity",
    )
    rougeL_recall_values = np.array(
        [evals["rougeL_recall"] for evals in scores_by_index.values()]
    )
    rougeL_recall_values = aggregate_to_1D(rougeL_recall_values)
    return {
        "agg_value": np.mean(rougeL_recall_values),
        "value_by_index": scores_by_index,
    }


def truth_ratio_helper(correct_answer_results, wrong_answer_results, aggregator):
    correct_indices = list(correct_answer_results.keys())
    correct_avg_losses = [
        evals["avg_loss"] for evals in correct_answer_results.values()
    ]
    wrong_indices = list(wrong_answer_results.keys())
    wrong_avg_losses = [evals["avg_loss"] for evals in wrong_answer_results.values()]

    assert correct_indices == wrong_indices
    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    truth_ratios = wrong_prob / correct_prob
    value_by_index = dict(
        zip(correct_indices, [{"truth_ratio": val} for val in truth_ratios])
    )
    truth_ratio_stats = np.array(
        [evals["truth_ratio"] for evals in value_by_index.values()]
    )
    forget_tr_avg = aggregator(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@unlearning_metric(name="forget_truth_ratio")
def forget_truth_ratio(model, **kwargs):
    """Compute the forget truth ratio (false/true), aggregating values using
    min(x, 1/x) to favor scores closer to 1 - which is ideal for forget set,
    and return the aggregated value."""

    def close_to_1(array):
        return np.mean(np.minimum(array, 1 / (array + 1e-10)))

    return truth_ratio_helper(
        kwargs["pre_compute"]["correct"]["value_by_index"],
        kwargs["pre_compute"]["wrong"]["value_by_index"],
        aggregator=close_to_1,
    )


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    """Compute the truth ratio (false/true), aggregating scores farther from 1
    (lower is better in general), and return the aggregated value."""

    def lower_from_1(array):
        return np.mean(np.maximum(0, 1 - array))

    return truth_ratio_helper(
        kwargs["pre_compute"]["correct"]["value_by_index"],
        kwargs["pre_compute"]["wrong"]["value_by_index"],
        aggregator=lower_from_1,
    )


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    values = [result["agg_value"] for _, result in kwargs["pre_compute"].items()]
    return {"agg_value": sc.stats.hmean(values)}
