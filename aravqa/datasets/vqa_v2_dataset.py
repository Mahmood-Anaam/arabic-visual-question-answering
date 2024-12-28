import torch

class VQAv2Dataset(torch.utils.data.Dataset):
    def __init__(self, BDS, VDS, language="ar"):
        self.BDS = BDS
        self.VDS = VDS
        self.language = language

    def __len__(self):
        return len(self.BDS)

    def __getitem__(self, idx):
        example = {
            "metadata": self.BDS[idx]["metadata"],
            "image": self.BDS[idx]["image"],
            "question": self.BDS[idx]["question"].get(self.language),
            "answers": self.BDS[idx]["answers"].get(self.language),
            "bit": {
                "captions": [cap.get("caption") for cap in self.BDS[idx]['captions']],
                "image_feats": self.BDS[idx]['features']['img_feats'],
                'od_labels': self.BDS[idx]['features']['od_labels']
            },
            "violet": {
              "captions": [cap.get("caption") for cap in self.VDS[idx]['captions']],
              "image_feats": self.VDS[idx]['features']
          },
            
        }
        return example