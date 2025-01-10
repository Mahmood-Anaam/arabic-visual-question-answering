from .okvqa_dataset import OKVQADataset

class VQAv2Dataset(OKVQADataset):
    def __init__(self, BDS, VDS):
        super().__init__(BDS=BDS,VDS=VDS)