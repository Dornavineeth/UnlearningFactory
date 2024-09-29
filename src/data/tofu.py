import datasets
import torch
from utils import package_prompt_response, add_dataset_index, get_model_cfg
from torch.utils.data import Dataset

class TOFU_QA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, split=None, question_key="question", answer_key="answer"):
        super(TOFU_QA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = 512
        self.data = datasets.load_dataset(data_path, split)["train"]
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_cfg(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        # indices = self.data[idx]["index"]
        if isinstance(answers, str):
            answers = [answers]

        input_ids_list = []
        label_list = []
        attention_mask_list = []

        for answer in answers:
            # apply chat template assuming model is chat model
            tokenized_data = package_prompt_response(self.model_configs, self.tokenizer,
                                                     question, answer, self.max_length)
            input_ids_list.append(tokenized_data['input_ids'])
            label_list.append(tokenized_data['labels'])
            attention_mask_list.append(tokenized_data['attention_mask'])

        return {
            'input_ids': torch.stack(input_ids_list).squeeze(),
            'labels': torch.stack(label_list).squeeze(),
            'attention_mask': torch.stack(attention_mask_list).squeeze(),
            # 'indices': torch.tensor(indices),
        }
