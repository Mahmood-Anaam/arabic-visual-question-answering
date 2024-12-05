from .field import RawField, Merge, ImagesField, TextField,ImagesField_noncoco
from .dataset import COCO, XM3600, CC3M
from torch.utils.data import DataLoader as TorchDataLoader



class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
