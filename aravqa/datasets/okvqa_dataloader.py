import torch
from aravqa.core.config import CaptionSelection
import numpy as np

class OKVQADataLoader:
    def __init__(self, dataset, config):
        """
        Initializes the data loader for the OKVQA dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset object.
            config (Config): The configuration object containing batch size and prompt formatting rules.
        """
        self.dataset = dataset
        self.batch_size = config.BATCH_SIZE
        self.device = config.DEVICE
        self.prompt_template = config.PROMPT_TEMPLATE
        self.captions_keys = config.CAPTIONS
        self.captions_separator = config.CAPTIONS_SEPARATOR
        self.caption_selection = config.CAPTION_SELECTION
        self.num_captions = config.NUM_CAPTIONS
        self.random_seed = config.RANDOM_SEED
    

    def filter_captions(self, captions, selection_strategy, num_captions):
        """
        Filters captions based on the selection strategy and the number of captions.

        Args:
            captions (List[Dict]): A list of caption dictionaries.
            selection_strategy (str): The caption selection strategy.
            num_captions (int): The maximum number of captions to include.

        Returns:
            List[str]: A filtered list of captions.
        
        """

        if num_captions > len(captions):
            raise ValueError(f"Number of captions ({num_captions}) exceeds the total number of captions ({len(captions)})")
        
        if (selection_strategy is None or 
            selection_strategy == CaptionSelection.NONE or 
            num_captions is None or 
            num_captions <= 0
            ):
            # No filtering; include all captions
            filtered_captions = captions
 
        elif selection_strategy == CaptionSelection.RANDOM:
            # Randomly select captions
            np.random.seed(self.random_seed)
            filtered_captions = np.random.choice(captions, num_captions, replace=False).tolist()
            
        elif selection_strategy == CaptionSelection.HIGH_SIMILARITY_QUESTION:
            # Sort captions by similarity to the question (descending) and select top-N
            filtered_captions = sorted(
                captions,
                key=lambda cap: cap.get("similarity_question", float('-inf')),
                reverse=True
            )[:num_captions]
        
        elif selection_strategy == CaptionSelection.LOW_SIMILARITY_QUESTION:
            # Sort captions by similarity to the question (ascending) and select top-N
            filtered_captions = sorted(
                captions,
                key=lambda cap: cap.get("similarity_question", float('inf'))
            )[:num_captions]
        
        elif selection_strategy == CaptionSelection.HIGH_SIMILARITY_ANSWERS:
            # Sort captions by similarity to the answers (descending) and select top-N
            filtered_captions = sorted(
                captions,
                key=lambda cap: cap.get("similarity_answers", float('-inf')),
                reverse=True
            )[:num_captions]
        
        elif selection_strategy == CaptionSelection.LOW_SIMILARITY_ANSWERS:
            # Sort captions by similarity to the answers (ascending) and select top-N
            filtered_captions = sorted(
                captions,
                key=lambda cap: cap.get("similarity_answers", float('inf'))
            )[:num_captions]
        
        elif selection_strategy == CaptionSelection.HIGH_SIMILARITY_QUESTION_ANSWERS:
            # Sort captions by similarity to both question and answers (descending) and select top-N
            filtered_captions = sorted(
                captions,
                key=lambda cap: cap.get("similarity_question", float('-inf')) + cap.get("similarity_answers", float('-inf')),
                reverse=True
            )[:num_captions]


        elif selection_strategy == CaptionSelection.LOW_SIMILARITY_QUESTION_ANSWERS:
            # Sort captions by similarity to both question and answers (ascending) and select top-N
            filtered_captions = sorted(
                captions,
                key=lambda cap: cap.get("similarity_question", float('inf')) + cap.get("similarity_answers", float('inf'))
            )[:num_captions]
           
        else:
            raise ValueError(f"Unknown caption selection strategy: {selection_strategy}")
        
        # Return only the "caption" field
        return [cap.get("caption", "") for cap in filtered_captions if cap.get("caption")]

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
        filtered_captions = []

        if self.captions_keys:
            for key in self.captions_keys:
                if key in captions:
                    # Filter captions based on the selection strategy and number of captions
                    selected_captions.extend(captions[key])
            
            filtered_captions.extend(self.filter_captions(selected_captions,
                                                          self.caption_selection,
                                                          self.num_captions
                                                          ))
            

     
                         
        # Join all selected captions
        formatted_captions = self.captions_separator.join(filtered_captions)
        formatted_captions = self.captions_separator + formatted_captions if formatted_captions else formatted_captions
        return self.prompt_template.format(question=question, context=formatted_captions)

    def collate_fn(self, batch):
        """
        Custom collate function for preparing the batch data.

        Args:
            batch (list): A list of samples from the dataset.

        Returns:
            dict: A dictionary containing batched prompts, answers, images, and metadata.
        """
        
        result = {
            "question_id":[],
            "image_id":[],
            "prompts":[],
            "answers":[],
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
