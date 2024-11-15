import torch
import logging
from torch import nn
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from collections import defaultdict

from evals.metrics.base import unlearning_metric

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)


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
        return next(iter(evals.values()))
    # for each index return a dict with all intra_item_idx values in list
    # now looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
    evals = dict_transpose(evals)
    print("Evaluated", len(evals), "indices")
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
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = evaluate_probability(model, dataloader)
    return scores_by_index


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = eval_text_similarity(
        model, tokenizer, dataloader, generation_args
    )
    return scores_by_index


@unlearning_metric(name="forget_truth_ratio")
def forget_truth_ratio(model, **kwargs):
    def aggregate_to_1D(x):
        return np.mean(x, axis=tuple(range(1, x.ndim)))

    correct_answer_results = kwargs["pre_compute"]["paraphrase"]
    wrong_answers_results = kwargs["pre_compute"]["perturb"]

    # Separate indices and losses
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
    forget_tr_stats = np.minimum(truth_ratios, 1 / (truth_ratios + 1e-10))
    forget_tr_avg = np.mean(forget_tr_stats)
    value_by_index = dict(
        zip(correct_indices, [{"truth_ratio": val for val in forget_tr_stats}])
    )
    value_by_index = dict(zip(correct_indices, forget_tr_stats.tolist()))

    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}
