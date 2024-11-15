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


def evaluate_probability_batch(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != -100).sum(-1)
    avg_loss = loss / num_token_gt
    normalized_probs = torch.exp(-avg_loss)
    return {'prob': normalized_probs.cpu().numpy().tolist(), 'avg_loss': avg_loss.cpu().numpy().tolist()}


def evaluate_probability(model, dataloader):
    evals = defaultdict(dict)
    for batch in tqdm(dataloader, desc="Calculating loss", total=len(dataloader)):
        # if data arrives in normal format we convert the batch to multiple answer-style
        # as in tofu_perturbed by adding a fake answer index
        if "input_ids" in batch:
            batch = {0: batch}
        # Assume batch like {"0": {"input_ids": [[]]..., "index": [453, 454..]}, 
        #                   "1": {.., "index":  [453, 454..]}..}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(iter(batch.values()))
        for intra_item_idx, mini_batch in batch.items():
            data_indices = mini_batch.pop("index").cpu().numpy().tolist() # data item indices
            batch_evals = evaluate_probability_batch(model, mini_batch) # batch_evals maps each attr to a list of length bsz
            transpose_batch_evals = [  # a list of len bsz which maps for each attr to an indiv value
                dict(zip(batch_evals.keys(), values)) 
                for values in zip(*batch_evals.values())
            ]
            indexwise_batch_evals = dict(zip(data_indices, transpose_batch_evals)) # map data indices to attr maps
            assert not (evals[intra_item_idx].keys() & indexwise_batch_evals.keys()), "Data indices repeated while iterating dataloader"
            evals[intra_item_idx] |= indexwise_batch_evals # append to existing collection for index
    
    if len(evals) == 1: # normal single answer dataset
        return evals[0]
    else:
        all_iidxs = list(evals.keys())
        all_idxs = list(evals[all_iidxs[0]].keys())
        all_stats = list(evals[all_iidxs[0]][all_idxs[0]].keys())
        # invert the dict, put outermost key to deepest
        evals = {idx: {stat: [evals[iidx][idx][stat] 
                              for iidx in all_iidxs]
                       for stat in all_stats}
                 for idx in all_idxs}
        return evals
    


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores["rouge1"].recall)
        rougeL_recall.append(rouge_scores["rougeL"].recall)
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def eval_text_similarity_batch(model, tokenizer, batch, generation_args):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    tokens = [label[label != -100] for label in labels]
    ground_truth = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    attention_mask = batch["attention_mask"]

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :], skip_special_tokens=True
    )
    scores = eval_rouge_recall(gen_text, ground_truth)
    scores.update(
        {"input": input_text, "ground_truth": ground_truth, "generation": gen_text}
    )
    # return list of dictionaries
    scores = [dict(zip(scores.keys(), values)) for values in zip(*scores.values())]
    return scores


def eval_text_similarity(model, tokenizer, dataloader, generation_args):
    index_to_scores = {}
    for batch in tqdm(
        dataloader, desc="Calculating Text Similarity", total=len(dataloader)
    ):
        if "input_ids" in batch:
            batch = {0: batch}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(
            iter(batch.values())
        )
        for _, mini_batch in batch.items():
            index = mini_batch.pop("index").numpy().tolist()
            scores = eval_text_similarity_batch(
                model, tokenizer, mini_batch, generation_args
            )
            assert len(index) == len(scores)
            for idx, score in zip(index, scores):
                if idx in index_to_scores:
                    if isinstance(index_to_scores[idx], list):
                        index_to_scores[idx].append(score)
                    else:
                        index_to_scores[idx] = [index_to_scores[idx]] + [score]
                else:
                    index_to_scores[idx] = score
    return index_to_scores


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
    def aggregate(x):
        return sum(x) / len(x) if isinstance(x, list) else x
    correct_answer_results = kwargs["pre_compute"]["paraphrase"]
    correct_loss = {idx: aggregate(result["avg_loss"])
                    for idx, result in correct_answer_results.items()}
    correct_prob = {idx: np.exp(-loss) for idx, loss in correct_loss.items()}
    
    wrong_answers_results = kwargs["pre_compute"]["perturb"]
    wrong_loss = {idx: aggregate(result["avg_loss"])
                  for idx, result in wrong_answers_results.items()}
    wrong_prob = {idx: np.exp(-loss) for idx, loss in wrong_loss.items()}
    
    assert correct_prob.keys() == wrong_prob.keys()
    truth_ratio = {idx: {"truth_ratio": wrong_prob[idx]/correct_prob[idx]}
                   for idx in correct_prob.keys()}
    return truth_ratio
