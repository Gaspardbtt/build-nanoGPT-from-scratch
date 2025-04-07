import os
import multiprocessing as mp
import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from google.colab import drive

# Monter Google Drive pour accéder à l'espace de stockage
drive.mount('/content/drive')

# Définir un répertoire dans Google Drive pour stocker les données
local_dir = '/content/drive/MyDrive/edu_fineweb10B'
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# Créer le répertoire local s'il n'existe pas encore
os.makedirs(local_dir, exist_ok=True)

# Télécharger le dataset

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# Initialiser le tokenizer

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # token spécial <|endoftext|> qui délimite les documents

def tokenize(doc):
    # Tokeniser un seul document et retourner un tableau numpy de tokens uint16
    tokens = [eot]  # le token spécial <|endoftext|> délimite tous les documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Le dictionnaire de tokens est trop grand pour uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Tokeniser tous les documents et écrire les morceaux de données, chaque morceau contenant shard_size tokens

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # Préallouer un tampon pour contenir le morceau actuel
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # Est-ce qu'il y a suffisamment d'espace dans le morceau actuel pour les nouveaux tokens ?
        if token_count + len(tokens) < shard_size:
            # Ajouter simplement les tokens au morceau actuel
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # Mettre à jour la barre de progression
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Écrire le morceau actuel et démarrer un nouveau
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}.npy")
            # Diviser le document pour ce qui rentre dans ce morceau ; le reste va dans le suivant
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # Remplir le prochain morceau avec les restes du document actuel
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # Écrire les tokens restants comme le dernier morceau
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])