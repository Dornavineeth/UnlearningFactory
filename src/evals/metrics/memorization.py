import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from collections import defaultdict

from evals.metrics.base import unlearning_metric


def dict_transpose(evals):
    # multiple answers indexed by intra_item_idx, then item_idx
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    # invert the dict, put outermost iidx deepest inside
    evals = {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }
    return evals


# Do you think helper functions like this and the above must be in some other file?
def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))


def run_batchwise_evals(model, dataloader, batch_eval_fn, batch_eval_fn_args, eval_msg):
    evals = defaultdict(dict)
    for batch in tqdm(dataloader, desc=eval_msg, total=len(dataloader)):
        # if data arrives in normal format we convert the batch to multiple answer-style
        # like in tofu_perturbed by adding a fake intra_item_index
        if "input_ids" in batch:
            batch = {"0": batch}
        # Assume batch like {"0": {"input_ids": [[]]..., "index": [453, 454..]},
        #                    "1": {"input_ids": [[]]..., "index": [453, 454..]}..}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(
            iter(batch.values())
        )
        for intra_item_idx, mini_batch in batch.items():
            data_indices = (
                mini_batch.pop("index").cpu().numpy().tolist()
            )  # data item indices
            batch_evals = batch_eval_fn(
                model=model, batch=mini_batch, **batch_eval_fn_args
            )
            indexwise_batch_evals = dict(zip(data_indices, batch_evals))
            assert not (
                evals[intra_item_idx].keys() & indexwise_batch_evals.keys()
            ), "Data indices repeated while iterating dataloader"
            evals[intra_item_idx] |= indexwise_batch_evals
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    if len(evals) == 1:  # normal single answer dataset, no need for list
        evals = next(iter(evals.values()))
    else:
        # for each index return a dict with all intra_item_idx values in list
        # now looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
        evals = dict_transpose(evals)
    print("Evaluated", len(evals), "examples")
    return evals


def evaluate_probability_batch(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # agg loss across tokens
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != -100).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    avg_losses = avg_losses.cpu().numpy().tolist()
    normalized_probs = normalized_probs.cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]


def evaluate_probability(model, dataloader):
    fun_args = {}
    return run_batchwise_evals(
        model, dataloader, evaluate_probability_batch, fun_args, "Calculating loss"
    )


def eval_text_similarity_batch(model, tokenizer, batch, generation_args):
    def eval_rouge_recall_batch(gen_outputs, ground_truths):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        evals = []
        for gen, gt in zip(gen_outputs, ground_truths):
            rouge_scores = scorer.score(gt, gen)
            evals.append(
                {
                    "rouge1_recall": rouge_scores["rouge1"].recall,
                    "rougeL_recall": rouge_scores["rougeL"].recall,
                }
            )
        return evals

    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    tokens = [label[label != -100] for label in labels]
    ground_truths = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    attention_mask = batch["attention_mask"]

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :], skip_special_tokens=True
    )
    scores = eval_rouge_recall_batch(gen_texts, ground_truths)
    scores = [
        {
            **rouge_evals,
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
        }
        for rouge_evals, input_text, ground_truth, gen_text in zip(
            scores, input_texts, ground_truths, gen_texts
        )
    ]
    return scores


def eval_text_similarity(model, tokenizer, dataloader, generation_args):
    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args}
    return run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity_batch,
        fun_args,
        "Calculating text similarity",
    )


@unlearning_metric(name="probability")
def probability(model, **kwargs):
    # returns the prob and avg_loss in scores
    # aggregate the prob values
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = evaluate_probability(model, dataloader)
    prob_values = np.array([evals["prob"] for evals in scores_by_index.values()])
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


@unlearning_metric(name="probability_w_options")
def probability_w_options(model, **kwargs):
    # normalises probability against that of false answers
    # needed for more open ended datasets
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
    # returns the rouge1_recall, rougeL_recall, input, ground_truth, generation keys in scores
    # aggregate the rougeL_recall values
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = eval_text_similarity(
        model, tokenizer, dataloader, generation_args
    )
    rougeL_recall_values = np.array(
        [evals["rougeL_recall"] for evals in scores_by_index.values()]
    )
    rougeL_recall_values = aggregate_to_1D(rougeL_recall_values)
    return {
        "agg_value": np.mean(rougeL_recall_values),
        "value_by_index": scores_by_index,
    }


def truth_ratio_helper(correct_answer_results, wrong_answers_results):
    correct_indices = list(correct_answer_results.keys())
    correct_avg_losses = [
        evals["avg_loss"] for evals in correct_answer_results.values()
    ]
    wrong_indices = list(wrong_answers_results.keys())
    wrong_avg_losses = [evals["avg_loss"] for evals in wrong_answers_results.values()]

    assert correct_indices == wrong_indices
    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    truth_ratios = wrong_prob / correct_prob
    value_by_index = dict(
        zip(correct_indices, [{"truth_ratio": val} for val in truth_ratios])
    )
    return value_by_index


@unlearning_metric(name="forget_truth_ratio")
def forget_truth_ratio(model, **kwargs):
    # returns truth_ratio value in indices (false/true)
    # aggregate by averaging min(x, 1/x) values of truth_ratio
    # because for forget truth ratio is better if closer to 1
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answers_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    value_by_index = truth_ratio_helper(correct_answer_results, wrong_answers_results)
    truth_ratio_stats = np.array(
        [evals["truth_ratio"] for evals in value_by_index.values()]
    )
    forget_tr_avg = np.mean(
        np.minimum(truth_ratio_stats, 1 / (truth_ratio_stats + 1e-10))
    )
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    # returns truth_ratio value in indices (false/true)
    # aggregate by averaging farther-from-1 values of truth_ratio
    # in general (false/true) truth ratio is better if lower
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answers_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    value_by_index = truth_ratio_helper(correct_answer_results, wrong_answers_results)
    truth_ratio_stats = np.array(
        [evals["truth_ratio"] for evals in value_by_index.values()]
    )
    forget_tr_avg = np.mean(np.maximum(0, 1 - truth_ratio_stats))
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}
