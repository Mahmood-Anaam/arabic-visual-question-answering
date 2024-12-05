
import torch
from transformers import AutoProcessor
import json
import h5py
from transformers import AutoTokenizer, GPT2TokenizerFast, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pickle
from light_normalizer import light_normalizer
import pandas as pd
class Coco_Dataset(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train"):
       self.img_root = img_root

       if split =="train":
        if ann_root is None:
                self.ann_root = "./annotations/NLLB_train_coco.json"
        else:
                 self.ann_root = str(ann_root)+"/NLLB_train_coco.json"
       else:
            if ann_root is None:
                    self.ann_root = "./annotations/NLLB_val_coco.json"
            else:
                    self.ann_root = str(ann_root)+"/NLLB_val_coco.json"
       self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14",cache_dir = "./")
       self.tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/Jasmine-350M")
       self.hf = h5py.File(self.img_root, 'r')
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
            self.data = pd.DataFrame(self.data)
    #    img_id = self.data[0]["image_id"]
    #    caption = self.data[0]["caption"]
    #    self.tokenizer.pad_token = "<|padding|>"
    #    self.tokenizer.bos_token = "<|endoftext|>"
    #    self.tokenizer.eos_token = "<|endoftext|>"
    #    self.tokenizer.add_bos_token = True
    #    normalizer = light_normalizer()
    #    breakpoint()
    #    cap_output = self.tokenizer(text= normalizer.run_light_normalizer(caption), padding='longest', add_special_tokens =True, return_tensors="pt")
    #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
    #    caption = self.processor(text= caption, return_tensors="pt")
   
    #    x =0


    def __getitem__(self, i):
        img_id = self.data.iloc[i]["image_id"]
        caption = self.data.iloc[i]["caption"]
        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, img_id

    def __len__(self):
        return len(self.data)

    
    def collate_fn(self, batch):
   
        images = torch.stack([example[0] for example in batch])
        ids = [[example[2] for example in batch]]
        normalizer = light_normalizer()
        self.tokenizer.pad_token = "<|padding|>"
        self.tokenizer.bos_token = "<|endoftext|>"
        self.tokenizer.eos_token = "<|endoftext|>"
        cap_output = self.tokenizer(text= [normalizer.run_light_normalizer("<|endoftext|> "+example[1]+" <|endoftext|>") for example in batch], padding='longest', add_special_tokens =True, return_tensors="pt")
        
        captions = cap_output["input_ids"]
        return images, captions, ids