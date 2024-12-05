import random
import wandb
from data import ImagesField, TextField, RawField,ImagesField_noncoco
from data import COCO,DataLoader,XM3600, CC3M
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Violet, VisualEncoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer
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
from transformers import AutoProcessor, CLIPVisionModelWithProjection
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



def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):


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


def evaluate_metrics(model, dataloader, text_field, exp_name=None, epoch=0):
    import itertools

    model.eval()

    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.module.beam_search(images, 20, text_field.vocab.stoi['<|endoftext|>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
                #gts['%d_%d' % (it, i)] = text_field.decode(gts_i[1::], join_words=False)
                #plt.imshow(np.transpose(images[i].cpu().detach().numpy(), (1, 2, 0)))
                #text_field.decode(gts_i[1::], join_words=False)

            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)



    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, text_field,gpt_optimizer,dataloader_eval,args):
    # Training with cross-entropy
    model.train()
    running_loss = .0

    model = DDP(model.module)
    model.to(device)
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):

            images, captions = images.to(device), captions.to(device)


            out,past= model(images, captions)

            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, 63999), captions_gt.view(-1)) #vocab size

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)

            gpt_optimizer.step()
            gpt_optimizer.zero_grad()


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
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--random_seed', type = int, default="42")
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--log_file',type = str, default="log/visualGPT.txt")
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
    

   # os.environ["WANDB_API_KEY"] = "your key"
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
        train_percentage = args.train_percentage,
        name = args.exp_name
    )
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    image_field = ImagesField(images_path=args.features_path, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<|endoftext|>', eos_token='<|endoftext|>', lower=True, 
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder,train_percentage=args.train_percentage,split_train_data=args.split_train_data)
    train_dataset, val_dataset, test_dataset = dataset.splits
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/Jasmine-350M")
    text_field.vocab = tokenizer.vocab



    # Model and dataloaders
    encoder = VisualEncoder(args.encoder_layer, 0, attention_module=ScaledDotProductAttention)
    model = Violet(text_field.vocab['<|endoftext|>'], encoder, args.decoder_layer,tau=args.tau)



    model = DDP(model)
    model.to(device)
    for name, param in model.named_parameters():

     if "h_lang" in name or "clip" in name and "visual_projection" not in name and "adapter" not in name  : #freeze language model and clip excpet for the projection head (and "visual_projection" not in name and "ln" not in name)

         param.requires_grad = False
    
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    ref_caps_train = list(train_dataset.text)

    cider_train=ref_caps_train
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))


    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})




    total_step_number = int(len(train_dataset)/(args.batch_size * args.gradient_accumulation_steps)*100)
 

    if args.optimizer_type =="adamw":
        
        gpt_optimizer = AdamW(model.module.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-8)
  
    elif args.optimizer_type =="adam":
        optimizer = Adam(model.module.parameters(), lr = args.lr)

 


    loss_fn = NLLLoss(ignore_index=text_field.vocab['<|padding|>'])
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
            gpt_optimizer.load_state_dict(data['optimizer'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    # use_rl=True
    # with wandb.init(mode="offline",project="Violet",config=config):
    #     wandb.watch(model,log="all", log_freq=1)
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                    drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,drop_last=True)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                        num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)


        train_loss = train_xe(model, dataloader_train, text_field,gpt_optimizer,dataloader_val,args)

        writer.add_scalar('data/train_loss', train_loss, e)

        # Validation loss

        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores

        scores = evaluate_metrics(model, dict_dataloader_val, text_field, args.exp_name+"_val", str(e))
        val_cider = scores['CIDEr']
        print("Cider score so far  "+str(scores['CIDEr']))
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        logging.info("val cider"+str(val_cider)+"current epoch "+str(e))
        logging.info("val bleu1" + str(scores["BLEU"][0]) + "current epoch " + str(e))
        logging.info("val bleu4" + str(scores["BLEU"][3]) + "current epoch " + str(e))
        logging.info("val rouge" + str(scores["ROUGE"]) + "current epoch " + str(e))



        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, args.exp_name+"_test", str(e))
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        logging.info("test cider" + str(scores['CIDEr']) + "current epoch " + str(e))
        logging.info("test bleu1" + str(scores["BLEU"][0]) + "current epoch " + str(e))
        logging.info("test bleu4" + str(scores["BLEU"][3]) + "current epoch " + str(e))
        logging.info("test meteor" + str(scores["METEOR"]) + "current epoch " + str(e))
        logging.info("test rouge" + str(scores["ROUGE"]) + "current epoch " + str(e))
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience +=1
            best = True
        else:
            patience = 0



        if patience == 30:
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
            'optimizer': gpt_optimizer.state_dict(),
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
            

#field.process
#PTBTokenizer.tokenize
