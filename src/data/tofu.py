from torch.utils.data import Dataset

from data.utils import load_hf_dataset, package_prompt_response, add_dataset_index


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answers = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answers, str):
            answers = [answers]

        items = []
        for answer in answers:
            # apply chat template assuming model is chat model
            tokenized_data = package_prompt_response(
                self.template_args,
                self.tokenizer,
                question,
                answer,
                self.max_length,
                self.predict_with_generate,
            )
            item_dct = {
                "input_ids": tokenized_data["input_ids"],
                "labels": tokenized_data["labels"],
                "attention_mask": tokenized_data["attention_mask"],
                "index": index,
            }
            items.append(item_dct)

        return items
