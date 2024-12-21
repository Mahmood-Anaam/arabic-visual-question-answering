'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import LayerNorm
from torch.nn import functional as F
from torch.nn import LayerNorm
from violet.modeling.transformer.containers import Module, ModuleList
from transformers import activations

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        test = x.contiguous().view(-1, x.size(-1))
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)

        x = x.view(*size_out)
        return x

class Attention_neo(Module):
    def __init__(self, nx, n_ctx, config, scale=False,can_be_stateful=False):
        super(Attention_neo, self).__init__()
        self.embed_dim = nx
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
       # self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.n_ctx= n_ctx
        self.scale = scale
        self.window_size= 256
        self.embed_dim = nx
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # self.c_attn = Conv1D(n_state * 3, nx)
        # self.c_proj = Conv1D(n_state, nx)
        self.can_be_stateful = can_be_stateful
        self.attn_pdrop = nn.Dropout(config.attn_pdrop)

        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((12,0, 64)))
            self.register_state('running_values', torch.zeros((12,0, 64)))


    def _attn(self, q, k, v,mask_self_attention,attention_type= "global"):
        bias = torch.tril(torch.ones((self.n_ctx, self.n_ctx), dtype=torch.uint8)).view(
             1, 1, self.n_ctx, self.n_ctx
        )
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, self.window_size)) #local attention window
        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1)) #Did not find it in Neo impelementation 

        if mask_self_attention is not None:


            w = w.masked_fill(mask_self_attention, -1e9)

        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_pdrop(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None,mask_self_attention=None):

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, key.transpose(-2,-1)],-2)
            key = self.running_keys.transpose(-2,-1)

            self.running_values = torch.cat([self.running_values, value], -2)
            value = self.running_values

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value,mask_self_attention)
        a = self.merge_heads(a)
        a = self.out_proj(a)


        return a, present

class Attention(Module):
    def __init__(self, nx, n_ctx, config, scale=False,can_be_stateful=False):
        super(Attention, self).__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
       # self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.n_ctx= n_ctx
        self.scale = scale
        self.window_size= 256
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.can_be_stateful = can_be_stateful
        self.attn_pdrop = nn.Dropout(config.attn_pdrop)

        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((12,0, 64)))
            self.register_state('running_values', torch.zeros((12,0, 64)))


    def _attn(self, q, k, v,mask_self_attention,attention_type= "global"):
        bias = torch.tril(torch.ones((self.n_ctx, self.n_ctx), dtype=torch.uint8)).view(
             1, 1, self.n_ctx, self.n_ctx
        )
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, self.window_size)) #local attention window
        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1)) #Did not find it in Neo impelementation 

        if mask_self_attention is not None:


            w = w.masked_fill(mask_self_attention, -1e9)

        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_pdrop(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None,mask_self_attention=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, key.transpose(-2,-1)],-2)
            key = self.running_keys.transpose(-2,-1)

            self.running_values = torch.cat([self.running_values, value], -2)
            value = self.running_values

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value,mask_self_attention)
        a = self.merge_heads(a)
        a = self.c_proj(a)


        return a, present


class Enc_Dec_Attention(Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Enc_Dec_Attention, self).__init__()
        n_state = nx = 768
        n_ctx = 60
        scale = True
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % 12 == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = 12
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

        self.fc_q = nn.Linear(n_state, 64 * 12)
        self.fc_k = nn.Linear(n_state, 64 * 12)
        self.fc_v = nn.Linear(n_state, 64 * 12)

        self.attn_dropout = nn.Dropout(0.2)

        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        # nn.init.xavier_uniform_(self.fc_o.weight)



    def _attn(self, q, k, v,enc_dec_attention):
        nk = k.shape[-1]
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        if enc_dec_attention is not None:
            w = w.masked_fill(enc_dec_attention, -10000.0)
            # w[:, :, ns-nd:ns, :ns] = w[:, :, ns-nd:ns, :ns].masked_fill(enc_dec_attention, -1e10)

        # w = w*enc_dec_attention

        # w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None,encoder_output=None, mask_encoder=None):

        query = self.fc_q(x)
        encoder_key = self.fc_k(encoder_output)
        encoder_value = self.fc_v(encoder_output)
        query = self.split_heads(query)
        encoder_key = self.split_heads(encoder_key, k=True)
        encoder_value = self.split_heads(encoder_value)


        a = self._attn(query, encoder_key,encoder_value,mask_encoder)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.adapter = Adapter(config)
        self.act = activations.get_activation("gelu_new")
        self.dropout = nn.Dropout(float(0.0))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        a = self.adapter(hidden_states)

        res = a + hidden_states 
        return res

class GPTNeoMLP_lang(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = activations.get_activation("gelu_new")
        

        self.dropout = nn.Dropout(float(0.0))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)



        return hidden_states





class Block(Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd

        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention_neo(nx, n_ctx, config, scale,can_be_stateful=True)
        self.enc_dec_attn = Enc_Dec_Attention(nx,n_ctx,config,scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(4 * nx, config) # project to higher dimension then bottleneck the MLP
        self.resid_pdrop= nn.Dropout(config.resid_pdrop)



        self.fc_alpha1 = nn.Linear(nx + nx, nx) 
        self.fc_alpha2 = nn.Linear(nx + nx, nx)
        self.fc_alpha3 = nn.Linear(nx + nx, nx)


    def forward(self,i, x, layer_past=None,mask_queries=None,encoder_output=None,mask_encoder=None, mask_self_attention=None, tau = 0):
        threshold = tau
        #layer_past is never used
        self_attention, present = self.attn(self.ln_1(x), layer_past=layer_past,
                                            mask_self_attention=mask_self_attention)
        a = x + self_attention
        a = self.resid_pdrop(a)


        enc_att1 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 0]),mask_encoder=mask_encoder) #mesh connection, mask_encoder=> padding masking
    
        enc_att2 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 1]),mask_encoder=mask_encoder) #mesh
    
        enc_att3 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 2]),mask_encoder=mask_encoder) #mesh
    

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([a, enc_att1], -1))) #Contrbution of each layer in the mesh connection
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([a, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([a, enc_att3], -1)))


        linguistics_alpha1_mask = torch.where(alpha1 > threshold, torch.ones_like(alpha1), torch.zeros_like(alpha1)) #Thresholding is on the alpha from M^2
        linguistics_alpha2_mask = torch.where(alpha2 > threshold, torch.ones_like(alpha2), torch.zeros_like(alpha2))
        linguistics_alpha3_mask = torch.where(alpha3 > threshold, torch.ones_like(alpha3), torch.zeros_like(alpha3))


        visual_alpha1_mask = torch.where(alpha1 < 1-threshold, torch.ones_like(alpha1), torch.zeros_like(alpha1))
        visual_alpha2_mask = torch.where(alpha2 < 1-threshold, torch.ones_like(alpha2), torch.zeros_like(alpha2))
        visual_alpha3_mask = torch.where(alpha3 < 1-threshold, torch.ones_like(alpha3), torch.zeros_like(alpha3))



        enc_att1 = alpha1* linguistics_alpha1_mask * a + (1-alpha1)* visual_alpha1_mask * enc_att1
        enc_att2 = alpha2* linguistics_alpha2_mask * a + (1-alpha2)* visual_alpha2_mask * enc_att2
        enc_att3 = alpha3* linguistics_alpha3_mask * a + (1-alpha3)* visual_alpha3_mask* enc_att3

        enc_att = (enc_att1 + enc_att2 + enc_att3) / np.sqrt(3)
        a = enc_att * mask_queries

        m = self.mlp(self.ln_2(a))

        encoder_result = a + m

        encoder_result = self.resid_pdrop(encoder_result)

        encoder_result = encoder_result  * mask_queries


        return encoder_result, present #output of each decoder layer goes to the input of the next layer

class Block_lang(Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block_lang, self).__init__()
        nx = config.n_embd

        # self.adapter = Adapter(config)

        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention_neo(nx, n_ctx, config, scale,can_be_stateful=True)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP_lang(4 * nx, config) # project to higher dimension then bottleneck the MLP
      #  self.resid_pdrop= nn.Dropout(config.resid_pdrop)
    def forward(self,i, x, layer_past=None,mask_queries=None,encoder_output=None,mask_encoder=None, mask_self_attention=None, tau = 0):
        #layer_past is never used
        self_attention, present = self.attn(self.ln_1(x), layer_past=layer_past,
                                            mask_self_attention=mask_self_attention)
        # self_adapter = self.adapter(self_attention) 
        # a = x + self_adapter
        # a = x 
        # a = self.resid_pdrop(a)
        #a = enc_att * mask_queries
        a = x + self_attention

        x = a

        a = self.mlp(self.ln_2(a))

        encoder_result = a + x
        # a = enc_att * mask_queries

        

        # encoder_result = a + m

        # encoder_result = self.resid_pdrop(encoder_result)

        # encoder_result = encoder_result  * mask_queries


        return encoder_result, present #output of each decoder layer goes to the input of the next layer

class Adapter(Module):
    def __init__(self,config):
        super(Adapter, self).__init__()
        nx = config.n_embd
        reduction_factor = 16
        self.down_sample_size = nx// reduction_factor
        self.activation = nn.SiLU()
        self.down_sampler = nn.Linear(nx, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, nx) 
    def forward(self,h):
        x = self.down_sampler(h)
        x = self.activation(x)
        x = self.up_sampler(x)
        return x
class GPT2Model(Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # block = Block(config.n_ctx, config, scale=True)
        self.h_lang = ModuleList([Block_lang(config.n_ctx, config, scale=True) for _ in range(config.n_layer//2)]) #only the first half
        self.h_cross = ModuleList([Block(config.n_ctx, config, scale=True) for _ in range((config.n_layer)//2)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.register_state('running_seq', torch.zeros((1,)).long())


    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None,mask_queries=None,encoder_output=None,mask_encoder=None, mask_self_attention = None, tau = 0):


        if past is None:
            past_length = 0
            past = [None] * (len(self.h_lang) + len(self.h_cross))
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []

        #language_decoder
        for i, (block, layer_past) in enumerate(zip(self.h_lang, past)):
            hidden_states, present = block(i,hidden_states, layer_past,mask_queries = mask_queries,encoder_output=encoder_output,mask_encoder=mask_encoder, mask_self_attention= mask_self_attention, tau = tau)
            presents.append(present)
        #cross_decoder
        for i, (block, layer_past) in enumerate(zip(self.h_cross, past)):
            hidden_states, present = block(i,hidden_states, layer_past,mask_queries = mask_queries,encoder_output=encoder_output,mask_encoder=mask_encoder, mask_self_attention= mask_self_attention, tau = tau)
            presents.append(present)     
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):

        lm_logits = self.decoder(hidden_state)
        return lm_logits



class GPT2LMHeadModel(Module):
    def __init__(self, config,padding_idx =1, tau = 0):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.padding_idx = padding_idx

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.tau = tau



    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, encoder_output=None, mask_encoder=None, position_ids=None, token_type_ids=None, lm_labels=None, past=None):

        b_s, seq_len = input_ids.shape[:2]
        mask_queries = (input_ids != self.padding_idx).unsqueeze(-1).float()

        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input_ids.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input_ids == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past,mask_queries=mask_queries,encoder_output=encoder_output,mask_encoder=mask_encoder, mask_self_attention= mask_self_attention, tau = self.tau)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss

        lm_logits = F.log_softmax(lm_logits,dim=-1)
        return lm_logits, presents
