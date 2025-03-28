import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


#-----------------------------------------------------------------------------------------------------------------------


# La classe CausalSelfAttention : 

# Implémente un mécanisme d'attention causale multi-têtes pour un modèle Transformer.  
# La classe projette l'entrée en queries, keys et values, applique une attention avec  
# un masque pour empêcher les tokens de voir le futur (causalité), puis combine les  
# résultats des différentes têtes avant de les projeter à nouveau. Cela permet au  
# modèle de capturer des dépendances contextuelles tout en maintenant un traitement  
# séquentiel adapté aux tâches de génération de texte.

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y  
    



# La classe MLP :
# Cette classe définit un Perceptron Multicouche (MLP) avec deux couches linéaires (fully connected).
# - La première couche (c_fc) transforme l'entrée de taille `config.n_embd` à `4 * config.n_embd`.
# - Une fonction d'activation GELU est appliquée à la sortie de la première couche.
# - La deuxième couche (c_proj) réduit la sortie de `4 * config.n_embd` à `config.n_embd`.
# Ce MLP est utilisé pour appliquer des transformations non linéaires aux embeddings dans un modèle de type Transformer.

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# La classe Block : 
# Ce bloc est une unité de base d'un Transformer (comme GPT).  
# Il applique :  
# 1. Une normalisation (LayerNorm)  
# 2. Une auto-attention causale avec connexion résiduelle  
# 3. Une deuxième normalisation  
# 4. Une MLP avec connexion résiduelle  
# Ces connexions résiduelles stabilisent l'entraînement et facilitent l'apprentissage.

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




#La classe GPTConfig est un dataclass qui stocke les paramètres de configuration du modèle :
#	•	block_size : Taille du contexte (longueur maximale d’une séquence d’entrée).
#	•	vocab_size : Nombre de tokens dans le vocabulaire (65 ici, ce qui est très petit, probablement pour un jeu de caractères ASCII).
#	•	n_layer : Nombre de couches de blocs Transformer.
#	•	n_head : Nombre de têtes dans l’attention multi-tête.
#	•	n_embd : Dimension des embeddings (taille des vecteurs d’entrée dans le modèle).

@dataclass
class GPTConfig:
    block_size: int = 1024               
    vocab_size: int =50257
    n_layer: int =12
    n_head: int =12
    n_embd: int = 768



# Classe GPT
# La classe GPT est un modèle basé sur les Transformers. Elle hérite de nn.Module et s’initialise avec la configuration config.

# a. Initialisation (__init__)
# 	•	self.config = config : Stocke la configuration du modèle.
# 	•	self.transformers est un ModuleDict contenant plusieurs sous-couches du Transformer :
# 	•	wte : Embedding pour les tokens (convertit les tokens en vecteurs denses).
# 	•	wpe : Embedding pour les positions (ajoute des informations de position à la séquence).
# 	•	h : Liste de Block(config), qui représentent les couches du Transformer (non définies ici mais probablement des blocs contenant des mécanismes d’attention et des MLP).
# 	•	ln_f : Normalisation LayerNorm à la fin du modèle.
# 	•	self.lm_head : Couche linéaire finale qui projette la sortie du Transformer vers l’espace du vocabulaire pour la prédiction des tokens suivants.


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        # idx is the shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # shape (T)
        pos_emb = self.transformer.wpe(pos)  #position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  #token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss




# La methode from_pretrained : 
# Charge un modèle GPT-2 pré-entraîné depuis Hugging Face et transfère ses poids  
# vers une implémentation personnalisée (GPT). Vérifie le type de modèle choisi,  
# initialise un modèle équivalent avec les bons paramètres, récupère les poids  
# depuis Hugging Face, adapte les différences de format (notamment pour les couches  
# de projection), et copie les poids dans le modèle personnalisé avant de le retourner.  
       
    @classmethod
    def from_pretrained(cls, model_type):
        """Charge les modèles pre-entrainés GPT-2 depuis huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    


#-----------------------------------------------------------------------------------------------------------------------

import tiktoken 

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # at init load tokens form disk and store them in memory for future training :) 
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state 
        self.current_position=0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]  
        x = (buf[:-1].view(B, T))  # inputs 
        y = (buf[1:].view(B, T))   # targets
        # advance the position in the tensor
        self.current_position += B*T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y



#-----------------------------------------------------------------------------------------------------------------------

import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"using device: {device}")


torch.manual_seed(13337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(13337)
# B=16 and T=256 for my GPU or it will blow up (T4 on Colab)
train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision('high')   # 'hight' --> matrice dot precision will be TensorFloat32 :) 
# get logits
model = GPT(GPTConfig())
model.to(device)  # Envoi du modèle sur l'appareil sélectionné
#logits, loss = model(x, y)


# Optimize!!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device) 
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000  #time diff en ms 
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f} ms, tok/sec: {tokens_per_sec:.2f}")




import sys; sys.exit(0)

# prefix tokens
num_return_sequences = 5
max_length = 30


enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long)   #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  #(5, 8)
x = tokens.to(device)  # Envoyer sur CPU ou GPU selon le cas

# generate ! right now x is (B, T) where B=5, T=8
# set seed to 42

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        #take the logits at the last position 
        logits = logits[:,-1,:] #(B, vocab_size)
        # get probabilities
        probs = F.softmax(logits, dim=-1)
        # do the top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  #(B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence 
        x = torch.cat((x, xcol), dim=1)


#print the generated text 
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)



