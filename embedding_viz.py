import torch
from torch import nn
import os, sys
import math
from torch.optim import Adam
from accelerate import Accelerator, notebook_launcher
import accelerate
import numpy as np
import pickle
from torch.utils.data import random_split
import tqdm
import random
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
sys.path.append(os.getcwd())
from Model.BERT import BERT, SMILESLM
import pandas as pd
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = 'cuda'

class SMILESDatasetPE(Dataset):
    def __init__(self, data, tokenizer, spe, seq_len, change_smi):
        self.data = data
        self.seq_len = seq_len
        self.len_compounds = len(self.data)
        self.tokenizer = tokenizer
        self.spe = spe
        self.change_smi = change_smi

    def __len__(self):
        return self.len_compounds

    def change_smiles(self, smi):
        if smi.find('.') >= 0:
            return smi

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        num_atoms = mol.GetNumAtoms()
        pos_list = list(range(num_atoms))

        pos = random.choice(pos_list)
        new_smi = Chem.MolToSmiles(mol, rootedAtAtom=pos)
        if len(new_smi) < self.seq_len:
            return new_smi
        else:
            return smi
        
    def __getitem__(self, item): 
        smiles = self.data.iloc[item]['SMILES']
        if self.change_smi:
            smiles = self.change_smiles(smiles)
        tokens = self.tokenizer.encode(self.spe.tokenize(smiles).split(' '))[0:-1][0:self.seq_len]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens = torch.tensor(tokens)

        y = self.data.iloc[item, 1:].values
        y = np.array(y, dtype=np.float32)
        y = torch.tensor(y)
    
        return [smiles, tokens], y

def prepare_data(data, tokenizer, spe, batch_size, seq_len, train_perc, num_workers, seed=10):
    #np.random.seed(seed)
    num_total = len(data)
    train_size = int(num_total * train_perc)
    
    indices = np.random.permutation(num_total)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    test_properties = data.iloc[test_indices,1:]
    test_properties = torch.tensor(test_properties.values, dtype=torch.float32)

    train_dataset = SMILESDatasetPE(data = data.iloc[train_indices], tokenizer = tokenizer, 
                                  spe = spe, seq_len = seq_len, change_smi = False)

    test_dataset = SMILESDatasetPE(data = data.iloc[test_indices], tokenizer = tokenizer, 
                                  spe = spe, seq_len = seq_len, change_smi = False)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, 
                                  num_workers=num_workers)

    return train_dataloader, test_dataset, len(train_dataset)


# data
smiles_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")
spe_vob = codecs.open('data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
# Load data
data = pd.read_csv("data/all4M_data.csv")

first_col = data.iloc[:, 0]
numeric_data = data.iloc[:, 1:]

Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)

# Remove columns where Q1 and Q3 are the same
numeric_data = numeric_data.loc[:, Q1 != Q3]

out_of_range = (numeric_data < -1e6) | (numeric_data > 1e6)
out_of_range_proportion = out_of_range.mean()
threshold = 0.5  
columns_to_keep = out_of_range_proportion[out_of_range_proportion <= threshold].index
numeric_data = numeric_data[columns_to_keep]
numeric_data = numeric_data[(numeric_data >= -1e6) & (numeric_data <= 1e6)].dropna()

# Recombine first column with cleaned numeric data
data = pd.concat([first_col.loc[numeric_data.index], numeric_data], axis=1)
num_props = len(data.columns)-1
chosen_descriptors = data.columns[1:]

print(f"Number of predicted molecular descriptors: {len(chosen_descriptors)}")

train_dataloader, testset, train_len  = prepare_data(data = data, tokenizer = smiles_tokenizer, spe = spe,
                                                                 batch_size = 1, seq_len = 100, 
                                                                 train_perc = 0.98, num_workers = 10, seed = 2)

test_loader = DataLoader(dataset = testset, shuffle=False, batch_size=1, num_workers=10)

# Model configuration
d_model = 512
n_layers = 4
heads = 8
dropout = 0.1
seq_length = 100

accelerator = Accelerator()
# Model configuration
vocab_size = len(smiles_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads,
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, output = 113)

smiles_model = accelerator.prepare(smiles_model)
accelerator.load_state(input_dir = "ckpts/checkpoint_19/")

bert_model.to(device)

embedding_list = []
smiles_list = []
properties_array = np.zeros((len(testset),len(chosen_descriptors)))

bert_model.eval()


with torch.no_grad():
    for batch, (X, Y) in tqdm.tqdm(enumerate(test_loader)):
        smiles = X[0]
        smiles = X[0] if isinstance(X[0], str) else X[0][0]
        X = X[1]
        X = X.to(device)
        Y = Y.to(device)
        smiles_list.append(smiles)
        embed = bert_model(X)
        mask = (X > 0).unsqueeze(-1)
        embeddings = embed * mask
        sum_embeddings = embeddings.sum(dim=1)  
        valid_tokens = mask.sum(dim=1)          
        mean_embeddings = sum_embeddings / valid_tokens
        embedding_list.append(mean_embeddings.cpu())
        properties_array[batch,:] = np.array([val.cpu() for val in Y])

properties_array = pd.DataFrame(data = properties_array, columns = numeric_data.columns)

flattened = [val for sublist in embedding_list for val in sublist]
embedding = np.array(flattened)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(embedding)

principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])

if not os.path.exists("figures/"):
    os.makedirs("figures/")

figures_dir = "figures/"

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

    plt.savefig(f'{figures_dir}{file_name}.png', dpi=400)
    plt.close(f)

else:
    plt.rcParams["figure.figsize"] = [4.50, 4.50]
    plt.rcParams["figure.autolayout"] = True
    f, ax = plt.subplots()
    plt.title("Embedding PCA")
    points = ax.scatter(principalDf[['PC1']], principalDf[['PC2']], s=20, alpha = 0.5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    file_name = "EmbeddingPCA"
    
    plt.savefig(f'{figures_dir}{file_name}.png', dpi=400)
    plt.close(f)








