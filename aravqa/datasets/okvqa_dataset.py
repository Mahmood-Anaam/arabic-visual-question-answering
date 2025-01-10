import torch

class OKVQADataset(torch.utils.data.Dataset):
    def __init__(self, BDS, VDS):
        self.BDS = BDS
        self.VDS = VDS
       

    def __len__(self):
        return len(self.BDS)

    def __getitem__(self, idx):
        example = {
            "metadata": self.BDS[idx]["metadata"],
            "image": self.BDS[idx]["image"],
            "question": self.BDS[idx]["question"],
            "answers": self.BDS[idx].get("answers",[]),
            "bit": self.BDS[idx].get("captions",[]),
            "violet": self.VDS[idx].get("captions",[])
            
        }
        return example