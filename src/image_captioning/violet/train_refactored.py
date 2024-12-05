import random
from dataset_refactored import Coco_Dataset
import wandb
from data import ImagesField, TextField, RawField,ImagesField_noncoco
from data import COCO,XM3600, CC3M
from torch.utils.data import DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Violet, VisualEncoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from sys import exit
import logging
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer
from transformers import AdamW
from torch import nn
# from accelerate import Accelerator
from datetime import datetime
from data.dataset import Dataset
# import pandas as pd
from torch.nn import DataParallel as DDP
from models.captioning_model import CaptioningModel
from PIL import Image
import glob
import json
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider
from transformers import AutoTokenizer
from light_normalizer import light_normalizer
def check_memory(cuda_device):
    """ Check the total memory and occupied memory for GPU """
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_memory(cuda_device):
    """ Create a large tensor and delete it.
    This operation occupies the GPU memory, so other processes cannot use the occupied memory.
    It is used to ensure that this process won't be stopped when it requires additional GPU memory.
    Be careful with this operation. It will influence other people when you are sharing GPUs with others.
    """
    for i,gpu in enumerate(cuda_device.split(',')):
        total, used = check_memory(gpu)
        cuda = torch.device('cuda:'+str(i))
        total = int(total)
        used = int(used)
        max_mem = int(total * 0.90)
        print('Total memory: ' + str(total) + ', used memory: ' + str(used))
        block_mem = max_mem - used
        if block_mem > 0:
            x = torch.FloatTensor(256, 1024, block_mem).to(device=cuda)
            del x



def evaluate_loss(model, dataloader, loss_fn):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions,_) in enumerate(dataloader):


                images, captions = images.to(device), captions.to(device)
                out,past = model(images, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, 63999), captions.view(-1)) #vocab size
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss
def load_references_from_json(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    captions_by_id = defaultdict(list)
    for item in data["annotations"]:
        captions_by_id[item['image_id']].append(item['caption'])
    
    return captions_by_id
def evaluate_cider(gen_captions, ref_captions):
    scorer = Cider()
    cider_score, _ = scorer.compute_score(ref_captions, gen_captions)
    print(f"CIDEr Score: {cider_score}")
    return cider_score

def evaluation(model, dataloader_val, ref_caps):
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/Jasmine-350M")
    model.eval()
    model = DDP(model.module)
    model = model.to("cuda")
    gen_caps = {}
    with tqdm( unit='it', total=len(dataloader_val)) as pbar:
        for it, (images, captions, ids) in enumerate(dataloader_val):
            images, captions = images.to("cuda"), captions.to("cuda")

            with torch.no_grad():
                out, _ = model.module.beam_search(images, 40, tokenizer.vocab['<|endoftext|>'], 5, out_size=1)
                generated_caption = tokenizer.batch_decode(out, skip_special_tokens=True)
                output = {key: [value] for key,value in zip(ids[0], generated_caption)}

            gen_caps = {**gen_caps, **output}
            pbar.update()
    ref_caps = dict(list(ref_caps.items())[0:len(gen_caps)])
    score = evaluate_cider(gen_caps, ref_caps)
    return score



def train_xe(model, dataloader,optimizer,dataloader_eval,args):
    # Training with cross-entropy
    
    model.train()
    running_loss = .0
    model = DDP(model.module)
    model.to(device)
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions,_) in enumerate(dataloader):

            images, captions = images.to(device), captions.to(device)


            out,past= model(images, captions)

            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, 63999), captions_gt.view(-1)) #vocab size

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            

            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


if __name__ == '__main__':
    now = datetime.now()

    current_time = now.strftime("%d-%b-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Violet')
    parser.add_argument('--exp_name', type=str, default='Violet'+str(current_time))
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--head', type=int, default=12)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--images_path', type=str, default="./coco_images.h5")
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--random_seed', type = int, default="42")
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--log_file',type = str, default="log/Violet.txt")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")


    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--optimizer_type', type= str, default = "adamw")
    parser.add_argument('--max_grad_norm', default=1.0, type = float)
    parser.add_argument('--train_percentage', default=1.0, type = float)
    parser.add_argument('--split_train_data', action="store_true")
    parser.add_argument("--decoder_layer", type= int, default = 12)
    parser.add_argument("--encoder_layer",type=int, default=3)
    parser.add_argument("--tau",type=float, default = 0.0)

    args = parser.parse_args()


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    

    #os.environ["WANDB_API_KEY"] = "add your key"
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
   # occupy_memory(os.environ["CUDA_VISIBLE_DEVICES"])
    n_gpus = torch.cuda.device_count()

    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info(args)
    #
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    config = dict(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        name = args.exp_name
    )
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Create the dataset
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/Jasmine-350M")
    # Model and dataloaders
    encoder = VisualEncoder(args.encoder_layer, 0, attention_module=ScaledDotProductAttention)
    model = Violet(tokenizer.vocab['<|endoftext|>'], encoder, args.decoder_layer,tau=args.tau)


    #using dataparallel, module is needed to access the model
    model = DDP(model)
    model.to(device)
    for name, param in model.named_parameters():

     if "h_lang" in name or "clip" in name and "visual_projection" not in name and "adapter" not in name and "ln" not in name  : #freeze language model and clip excpet for the projection head and adapter

         param.requires_grad = False
    
 

    if args.optimizer_type =="adamw":
        
        optimizer = AdamW(model.module.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-8)
  
    elif args.optimizer_type =="adam":
        optimizer = Adam(model.module.parameters(), lr = args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor = 0.5)


    loss_fn = NLLLoss(ignore_index=tokenizer.vocab['<|padding|>'])
    use_rl = False
    best_cider = .0
    best_loss = np.inf
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optimizer.load_state_dict(data['optimizer'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    #### Uncomment to use wandb and adjust indentation
    # with wandb.init(mode="offline",project="Violet",config=config):
    #     wandb.watch(model,log="all", log_freq=1)
    dataset_train = Coco_Dataset(img_root = args.images_path)
    dataset_val = Coco_Dataset(img_root = args.images_path, split="val")

    ref_caps = load_references_from_json("./annotations/NLLB_val_coco.json")

    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,collate_fn = dataset_train.collate_fn,
                                    drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn = dataset_val.collate_fn, drop_last=True)



        train_loss = train_xe(model, dataloader_train,optimizer,dataloader_val,args)

        writer.add_scalar('data/train_loss', train_loss, e)

        # Validation loss

        val_loss = evaluate_loss(model, dataloader_val, loss_fn)
        scheduler.step(val_loss)
        writer.add_scalar('data/val_loss', val_loss, e)

        val_cider = evaluation(model, dataloader_val, ref_caps)
        # Validation scores


        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience +=1
            best = True
        else:
            patience = 0



        if patience == 10:
            break
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)
        # wandb.log({"Cider score  ": val_cider})
        # wandb.log({"train_loss  ": train_loss})
        # wandb.log({"loss_val  ": val_loss})
        # wandb.log({"BLEU4 score  ": scores['BLEU'][3]})
        # wandb.log({"ROUGE score  ": scores['ROUGE']})
        

