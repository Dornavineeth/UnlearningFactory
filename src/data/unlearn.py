import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, fix="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            fix (str, optional): Specifies which dataset to fix while sampling from the other. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.fix = fix

    def __len__(self):
        if self.fix == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when fix=forget"
            )
            return len(self.forget)
        elif self.fix == "retain":
            assert self.forget is not None, ValueError(
                "retain dataset can't be None when fix=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.fix} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.fix == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
        elif self.fix == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
        return item
