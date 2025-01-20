from .okvqa_dataset import OKVQADataset

class VQAv2Dataset(OKVQADataset):
    def __init__(self, BDS, VDS):
        super().__init__(BDS=BDS,VDS=VDS)

    def __getitem__(self, idx):
        example = {
            "metadata": self.BDS[idx]["metadata"],
            "image": self.BDS[idx]["image"],
            "question": self.BDS[idx]["question"],
            "answers": self.BDS[idx].get("answers",[]),
            "multiple_choice_answer":self.BDS[idx]["multiple_choice_answer"],
            "bit": self.BDS[idx].get("captions",[]),
            "violet": self.VDS[idx].get("captions",[])
            
        }
