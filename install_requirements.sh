#!/bin/bash

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "Installation des bibliothèques nécessaires..."

pip install torch torchvision torchaudio
pip install tiktoken
pip install numpy
pip install torchdistributed
pip install tqdm
pip install datasets
pip install transformers

# Vérifier si les bibliothèques sont installées correctement
echo "Vérification de l'installation des bibliothèques..."

python -c "import torch; import transformers; import tiktoken; import tqdm; import datasets; import numpy; import torch.distributed as dist; print('Tous les packages sont installés correctement')"

echo "Installation terminée."


# chmod +x install_requirements.sh pour rendre executable 

# ./install_requirements.sh pour executer le script