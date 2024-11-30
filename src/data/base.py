# import torch
from torch.utils.data import Dataset
from data.utils import load_hf_dataset, add_dataset_index, package_prefix_cont


class ContinuationDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        continuation_key="gt",
        max_cont_len=128,
        predict_with_generate=False,
    ):
        super(ContinuationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_cont_len = max_cont_len
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.prefix_key = prefix_key
        self.continuation_key = continuation_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, continuation, index=-1):
        tokenized_data = package_prefix_cont(
            self.tokenizer,
            prefix,
            continuation,
            self.max_cont_len,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx][self.prefix_key]
        cont = self.data[idx][self.continuation_key]
        index = self.data[idx]["index"]
        item = self._process_sample(pref, cont, index)
        return item
