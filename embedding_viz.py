import argparse
import torch
from torch import nn
import os, sys
import math
from accelerate import Accelerator, notebook_launcher
import accelerate
import numpy as np
import pickle
import random
import datetime
from transformers import BertTokenizer
import torch.nn.functional as F
#from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import tqdm
import pandas as pd
from sklearn.decomposition import PCA
import json

sys.path.append(os.getcwd())
from Model.BERT import BERT, SMILESLM

import re

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

class BasicSmilesTokenizer(object):
    """
    Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al.
    This tokenizer is to be used when a tokenizer that does not require the transformers library by HuggingFace is required.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
    >>> tokenizer = BasicSmilesTokenizer()
    >>> print(tokenizer.tokenize("CC(=O)OC1=CC=CC=C1C(=O)O"))
    ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', '(', '=', 'O', ')', 'O']


    References
    ----------
    .. [1] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
        ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
        1572-1583 DOI: 10.1021/acscentsci.9b00576
    """

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        """Constructs a BasicSMILESTokenizer.

        Parameters
        ----------
        regex: string
            SMILES token regex
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        """Basic Tokenization of a SMILES.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


def custom_list(values):
    if values is None:
        return ["H_bond_acceptor", "H_bond_donor", "Rotatable_bond", "Exact_mass", "TPSA", "Heavy_atom LogP"]
    if isinstance(values, list):
        return values
    return values.split(',')

parser = argparse.ArgumentParser()

parser.add_argument("--cpu", help="Use CPUs", action="store_true")
parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to validate")
parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
parser.add_argument("--data_path", type=str, default="/data/smiles_string.txt", help="Path to the index file")
parser.add_argument("--results_path", type=str, default="/home/ubuntu/BertSmiles/results/", help="Directory to save the results generated.")
parser.add_argument("--properties_names", type=custom_list, nargs='*', 
                    help="List of properties names. If no properties are present in the data, call --properties_names without any value. If the default properties list wants to be used, do not call --properties_names.")

args = parser.parse_args()


with open(args.data_path, 'r') as file:
    smiles_list = json.load(file)

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, properties_names, seq_len):
        self.smiles_list = smiles_list
        self.seq_len = seq_len
        self.len_compounds = len(self.smiles_list)
        self.properties_names = properties_names

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):   
        if len(self.properties_names)>0:
            smiles = self.smiles_list[item][0]
            basictokenizer = BasicSmilesTokenizer(SMI_REGEX_PATTERN)
            tokens = tokenizer.encode(basictokenizer.tokenize(smiles))[0:-1]
            padding = [tokenizer.encode('[PAD]')[1] for _ in range(384 - len(tokens))]
            tokens.extend(padding)
            #pred = smiles_model(tokens)
            tokens = torch.tensor(tokens)
            properties = torch.tensor(self.smiles_list[item][1:])
        
        else:
            smiles = self.smiles_list[item]
            basictokenizer = BasicSmilesTokenizer(SMI_REGEX_PATTERN)
            tokens = tokenizer.encode(basictokenizer.tokenize(smiles))[0:-1]
            padding = [tokenizer.encode('[PAD]')[1] for _ in range(384 - len(tokens))]
            tokens.extend(padding)
            #pred = smiles_model(tokens)
            tokens = torch.tensor(tokens)
            properties = torch.tensor(0)
    
        return tokens, properties

## Auxiliar functions
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.enabled = False
    print(f"Random seed set as {seed}")
    
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(1901)


def prepare_data(smiles_list, tokenizer, properties_names,
                 seq_len, num_workers, seed = 10):
    
    smiles_dataset = SMILESDataset(smiles_list = smiles_list, 
                                   properties_names = properties_names,
                                   tokenizer = tokenizer, 
                                   seq_len = seq_len)
    
    train_dataloader = DataLoader(dataset = smiles_dataset, shuffle=True, batch_size=1,
                                  num_workers=num_workers, worker_init_fn=seed_worker)
    
    #eval_dataloader = DataLoader(dataset = testset, shuffle=False, batch_size=batch_size, num_workers=4)
    
    return train_dataloader

### Data
# Hbond acceptor, Hbond donor, Rotatable bond, Exact mass, TPSA, Heavy atom count, Log-P
# data
tokenizer = BertTokenizer("data/atomlevel_tokenizer/vocab.txt")

d_model = 90
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 384
batch_size=1

if args.cpu:
    device = 'cpu'
else:
    device = 'cuda'

accelerator = Accelerator()

# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model)
smiles_model.to(device)

# data preparation
train_loader = prepare_data(smiles_list = smiles_list, tokenizer = tokenizer, 
                            properties_names = args.properties_names,
                            seq_len = seq_length, num_workers = args.num_workers, seed = 2)


# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/atomlevel_ckp/")


smiles_model.eval()

loss_fn = nn.L1Loss().to(device)

if len(args.properties_names)>0:
    properties_names = args.properties_names
    properties_array = np.zeros((args.n_samples,len(properties_names)))

embedding_list = []


print('validating...')
#  preventing gradient calculations since we will not be optimizing
with torch.no_grad():
    for batch, (X, Y) in tqdm.tqdm(enumerate(train_loader)):
        if batch == args.n_samples:
            break
            
        X = X.to(device)
        Y = Y.to(device)
        
        pred,embed = smiles_model(X)

        embedding_list.append(embed.cpu())
        
        if len(args.properties_names)>0:
            properties_array[batch,:] = np.array([val.cpu() for val in Y])

if len(args.properties_names)>0:
    properties_array = pd.DataFrame(data = properties_array, 
                                columns = properties_names)


flattened = [val for sublist in embedding_list for val in sublist]
embedding = np.array(flattened)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(embedding)

principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])

figures_dir = f'{args.results_path}/figures/'

if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

if len(args.properties_names)>0:
    finalDf = pd.concat([principalDf, properties_array], axis = 1)

    for n in properties_array.columns:
        plt.rcParams["figure.figsize"] = [4.50, 4.50]
        plt.rcParams["figure.autolayout"] = True
        f, ax = plt.subplots()
        plt.title(n)
        points = ax.scatter(finalDf[['PC1']], finalDf[['PC2']], c=np.array(finalDf[n]), s=20, 
                            cmap="plasma", alpha = 0.5)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        f.colorbar(points)
    
        file_name = n.replace(" ", "_").replace("-", "_")
        
        plt.savefig(f'{figures_dir}{file_name}.png')

else:
    plt.rcParams["figure.figsize"] = [4.50, 4.50]
    plt.rcParams["figure.autolayout"] = True
    f, ax = plt.subplots()
    plt.title("Embedding PCA")
    points = ax.scatter(principalDf[['PC1']], principalDf[['PC2']], s=20, alpha = 0.5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    file_name = "EmbeddingPCA"
    
    plt.savefig(f'{figures_dir}{file_name}.png')

    
smiles_embed = {smiles: matrix for smiles, matrix in zip(smiles_list, embedding_list)}
np.savez(f'{args.results_path}/embeddings.npz', **smiles_embed)
print("Embeddings saved to", f'{args.results_path}/embeddings.npz')






