from torch.utils.data import DataLoader
import torch

class OKVQADataLoader:
    """
    DataLoader wrapper for the OKVQADataset.
    Provides easy integration with the dataset and supports batching, shuffling, and parallel loading.
    """

    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True):
        """
        Initializes the DataLoader for the OKVQADataset.

        Args:
            dataset (Dataset): Instance of OKVQADataset.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset at every epoch.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): Whether to pin memory (for CUDA optimization).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Initialize PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn  # Custom collate function for flexible batching
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle flexible data batching.

        Args:
            batch (list): List of samples fetched from the dataset.

        Returns:
            dict: A batch dictionary containing all data merged into tensors.
        """
        # Separate indices and examples
        indices, examples = zip(*batch)

        inputs = {
                "input_ids": torch.stack([torch.tensor(example["inputs"]["input_ids"]) for example in examples]),
                "attention_mask": torch.stack([torch.tensor(example["inputs"]["attention_mask"]) for example in examples]),
                "token_type_ids": torch.stack([torch.tensor(example["inputs"]["token_type_ids"]) for example in examples]),
                "img_feats": torch.stack([torch.tensor(example["inputs"]["img_feats"]) for example in examples]),
                "masked_pos": torch.stack([torch.tensor(example["inputs"]["masked_pos"]) for example in examples]),
            }

        # Merge examples into a single batch dictionary
        batch_dict = {
            "indices": indices,  # List of indices
            # "metadata": [example["metadata"] for example in examples],  # Metadata for each sample
            # "images": [example["image"] for example in examples],  # Raw PIL images
            "questions": [example["question"] for example in examples],  # Tokenized questions
            "answers": [example["answers"] for example in examples],  # List of answers
            # "raw_answers": [example["raw_answers"] for example in examples],  # Raw answers
            "confidence_answers": [example["confidence_answers"] for example in examples],  # Answer confidences
            # "object_detections": [example["object_detections"] for example in examples],  # Object detection details
            "inputs": inputs  # Batched model inputs as tensors

        }

        return batch_dict

    def __iter__(self):
        """
        Creates an iterator over the DataLoader.

        Returns:
            iterator: Iterator over batches of the dataset.
        """
        return iter(self.dataloader)

    def __len__(self):
        """
        Returns the number of batches in the DataLoader.

        Returns:
            int: Total number of batches.
        """
        return len(self.dataloader)