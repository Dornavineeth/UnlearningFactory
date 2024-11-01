from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain):
        self.forget = forget
        self.retain = retain

    def __len__(self):
        if self.forget:
            return len(self.forget)
        else:
            return len(self.retain)

    def __getitem__(self, idx):
        item = {}
        if self.forget:
            item["forget"] = self.forget[idx]
        if self.retain:
            item["retain"] = self.retain[idx]
        return item
