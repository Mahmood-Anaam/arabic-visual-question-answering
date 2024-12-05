import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from models.transformer.decoder import GPT2LMHeadModel
from models.transformer.config import GPT2Config
from transformers import AutoTokenizer, AutoModelForCausalLM




class Violet(CaptioningModel):

    def __init__(self, bos_idx, encoder,n_layer=12,tau=0):
        super(Violet, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir = "./").to("cuda")
        # self.clip.to("cuda")
        jasmine = AutoModelForCausalLM.from_pretrained("UBC-NLP/Jasmine-350M")  
        state_dict = jasmine.state_dict()
        config = GPT2Config()
        decoder = GPT2LMHeadModel(config,tau=tau)

        #loading jasmine state dict
        names = []
        for name, param in decoder.named_parameters():
            #choose only jasmine layers
            if "enc" not in name and "alpha" not in name and "adapter" not in name:

                names.append(name)
        
        filterd_state_dict = {}
        #remove unmatching keys
        for key, value in state_dict.items():
            if "masked" in key or "lm_head" in key or "attn.attention.bias" in key:
                continue
            filterd_state_dict[key] = value 




        new_state_dict = {}
        for name, (key, value) in zip(names,filterd_state_dict.items()):
            # Map key to new layer name
            new_state_dict[name] = value
        # decoder = load_weight(decoder, state_dict)
        decoder.load_state_dict(new_state_dict,strict=False)
        decoder.set_tied()
                    
        self.decoder = decoder


        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()
        #commit
    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        
            for p in self.encoder.parameters():
                if p.dim()> 1:
                    nn.init.xavier_uniform_(p)




    def forward(self, images, seq, *args):
        #enc_output, mask_enc = self.encoder(images) #mask encoder is not important, it was for empty detections
        #model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        images = images.to("cuda")
        outputs = self.clip(images) #the clip boi
        image_embeds = outputs.image_embeds # Visual projection output
        image_embeds = image_embeds.unsqueeze(1)
        # image_embeds = outputs.last_hidden_state #patches
        enc_output,_ = self.encoder(image_embeds) #Three encoders output
        dec_output,past = self.decoder(seq, enc_output)
        return dec_output,past

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, past, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                with torch.no_grad():
                    outputs = self.clip(visual) #the clip boi
                    image_embeds = outputs.image_embeds # Visual projection output
                    image_embeds = image_embeds.unsqueeze(1)
                self.enc_output, self.mask_enc = self.encoder(image_embeds)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc,past=past)

