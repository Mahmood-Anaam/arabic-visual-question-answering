import torch

class VQAv2Dataset(torch.utils.data.Dataset):
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
            "multiple_choice_answer": self.BDS[idx]["multiple_choice_answer"],
            "bit": self.BDS[idx].get("captions",[]),
            "violet": self.VDS[idx].get("captions",[])
            
        }
        return example