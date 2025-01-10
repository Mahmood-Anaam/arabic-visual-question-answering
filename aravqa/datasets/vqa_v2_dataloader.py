from .okvqa_dataloader import OKVQADataLoader

class VQAv2DataLoader(OKVQADataLoader):
    def __init__(self, dataset, config):
        """
        Initializes the data loader for the VQA_v2 dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset object.
            config (Config): The configuration object containing batch size and prompt formatting rules.
        """
        super().__init__(dataset=dataset,config=config)


