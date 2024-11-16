import torch
import logging
from torch import nn
from tqdm import tqdm
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

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
    return normalized_probs.cpu().numpy().tolist()


def evaluate_probability(model, dataloader):
    index_to_prob = {}
    for batch in tqdm(dataloader, desc="Calculating loss", total=len(dataloader)):
        if "input_ids" in batch:
            batch = {0: batch}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(iter(batch.values()))
        for _, mini_batch in batch.items():
            index = mini_batch.pop("index").cpu().numpy().tolist()
            probs = evaluate_probability_batch(model, mini_batch)
            for idx, prob in zip(index, probs):
                if idx in index_to_prob:
                    if isinstance(index_to_prob[idx], list):
                        index_to_prob[idx].append({"prob": prob})
                    else:
                        index_to_prob[idx] = [index_to_prob[idx]] + [{"prob": prob}]
                else:
                    index_to_prob[idx] = {"prob": prob}
    return index_to_prob


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
    index_to_probs = evaluate_probability(model, dataloader)
    return index_to_probs


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    index_to_scores = eval_text_similarity(
        model, tokenizer, dataloader, generation_args
    )
    return index_to_scores


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    para_results = kwargs["pre_compute"]["paraphrase"]
    pert_results = kwargs["pre_compute"]["perturb"]
    index_to_scores = {}
    for k, para_result in para_results.items():
        para_prob = para_result["prob"]
        pert_result = pert_results[k]
        pert_prob = sum([r["prob"] for r in pert_result]) / len(pert_result)
        index_to_scores[k] = {"truth_ratio": pert_prob / para_prob}
    return index_to_scores
