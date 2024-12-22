import torch
from llms.configuration import Config

import torch

class VQAv2DataLoader:
    def __init__(self, dataset, config=Config):
        """
        Initializes the data loader for the VQAv2 dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset object.
            config (Config): The configuration object containing batch size and prompt formatting rules.
        """
        self.dataset = dataset
        self.batch_size = config.BATCH_SIZE
        self.device = config.DEVICE
        self.prompt_template = config.PROMPT_TEMPLATE  
        self.captions_keys = config.CAPTIONS
        

    def create_prompt(self, question, captions):
        """
        Generates a prompt based on the provided question and captions using the template from Config.

        Args:
            question (str): The question text.
            captions (dict): A dictionary containing captions for different models.

        Returns:
            str: The formatted prompt.
        """
        # Filter and format captions based on the specified keys in Config
        selected_captions = []
        if self.captions_keys:
          for key in self.captions_keys:
              if key in captions:
                  selected_captions.extend(captions[key])
        
        formatted_captions = "\n".join(selected_captions)
        return self.prompt_template.format(question=question, captions=formatted_captions)

    def collate_fn(self, batch):
        """
        Custom collate function for preparing the batch data.

        Args:
            batch (list): A list of samples from the dataset.

        Returns:
            dict: A dictionary containing batched prompts, answers, images, and metadata.
        """
        prompts = []
        answers = []
        images = []
        metadata = []

        for item in batch:
            prompt = self.create_prompt(item["question"], {
                "bit": item["bit"]["captions"],
                "violet": item["violet"]["captions"]
            })
            prompts.append(prompt)
            answers.append(item["answers"])
            images.append(item["image"])
            metadata.append(item["metadata"])

        return {
            "prompts": prompts,
            "answers": answers,
            "images": images,
            "metadata": metadata
        }

    def get_dataloader(self):
        """
        Returns a DataLoader object for the dataset.

        Returns:
            torch.utils.data.DataLoader: The DataLoader instance.
        """
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )