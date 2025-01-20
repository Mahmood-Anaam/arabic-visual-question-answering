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

     def collate_fn(self, batch):
        """
        Custom collate function for preparing the batch data.

        Args:
            batch (list): A list of samples from the dataset.

        Returns:
            dict: A dictionary containing batched prompts, answers, images,multiple_choice_answer, and metadata.
        """
        
        result = {
            "question_id":[],
            "image_id":[],
            "prompts":[],
            "answers":[],
            "multiple_choice_answer":[],
        }
        

        for item in batch:
            # Create a prompt using filtered captions
            prompt = self.create_prompt(item["question"], {
                "bit": item["bit"],
                "violet": item["violet"]
            })
            
            result["question_id"].append(item["metadata"]["question_id"])
            result["image_id"].append(item["metadata"]["image_id"])
            result["prompts"].append(prompt)
            result["answers"].append(list(map(lambda ans:ans.get("answer",""),item["answers"])))
            result["multiple_choice_answer"].append(item["multiple_choice_answer"])
            
        return result
                                     

     

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


