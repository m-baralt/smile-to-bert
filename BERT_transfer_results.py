######################
### import modules ###
######################
import pandas as pd
import numpy as np
import sys
import torch
from transformers import BertTokenizer
from accelerate import Accelerator
import os
sys.path.append(os.getcwd())
from Model.BERT import BERT, SMILESLM
from Model.binding_functions import FC, FCT, SMILESDataset, SMILESDatasetT, SMILESDatasetPE, SMILESDatasetJ, CosineWithRestarts
from Model.training_funcs import train, trainT, trainF, trainPE, trainPE_large, trainJ
from Model.Transformer import ALPHABET_SIZE, Transformer
from Model.finetuned_transformer import ALPHABET_SIZE, TransformerBert
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer

os.makedirs("Experiments_results")

#####################################
### Prepare BERT model - atomwise ###
#####################################

tokenizer = BertTokenizer("data/atomlevel_tokenizer/vocab.txt")
device = "cuda"
d_model = 90
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 384
batch_size=1

accelerator = Accelerator()

# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/atomlevel_ckp/")


##########################################
### Prepare data - dataset AID 1053197 ###
##########################################

# undersample for 1053197
def undersample(idxs, labels, ratio=1):
    no_bind_idxs = idxs[labels[idxs]==0]
    bind_idxs = idxs[labels[idxs]==1]
    min_len = min(len(bind_idxs), len(no_bind_idxs)) * ratio
    
    np.random.shuffle(no_bind_idxs)
    np.random.shuffle(bind_idxs)
    
    no_bind_idxs = no_bind_idxs[:min_len]
    bind_idxs = bind_idxs[:min_len]
    
    idxs = np.concatenate((no_bind_idxs, bind_idxs))
    np.random.shuffle(idxs)
    
    return idxs

# AID 1053197
aid1053197 = pd.read_csv('experiments/1053197.tsv', sep='\t')
binding = aid1053197['binding']
idxs = undersample(np.arange(len(binding)), binding, ratio=1)

aid1053197 = aid1053197.iloc[idxs]

dataset = aid1053197

model = FC(input_features=90*83)

cv_accs, cv_labs, cv_preds = train(embedding_model = smiles_model, tokenizer = tokenizer,
                                   dataset = dataset, seq_len = 384, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_1053197.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)



##########################################
### Prepare data - dataset AID 652067 ###
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/atomlevel_ckp/")

# undersample for 652067
def undersample(idxs, labels, ratio=1):
    no_bind_idxs = idxs[labels[idxs] == 0]
    bind_idxs = idxs[labels[idxs] == 1]
    min_len = min(len(bind_idxs), len(no_bind_idxs)) * ratio
    
    np.random.shuffle(no_bind_idxs)
    np.random.shuffle(bind_idxs)
    
    no_bind_idxs = no_bind_idxs[:min_len]
    bind_idxs = bind_idxs[:min_len]
    
    idxs = np.concatenate((no_bind_idxs, bind_idxs))
    total_len = len(idxs)
    
    # Adjust the length to be the largest multiple of 10 less than or equal to current length
    adjusted_len = (total_len // 10) * 10
    
    np.random.shuffle(idxs)
    idxs = idxs[:adjusted_len]
    
    return idxs

# AID 652067
aid652067 = pd.read_csv('experiments/652067.tsv', sep='\t')
binding = aid652067['binding']
idxs = undersample(np.arange(len(binding)), binding, ratio=1)

aid652067 = aid652067.iloc[idxs]

dataset = aid652067

cv_accs, cv_labs, cv_preds = train(embedding_model = smiles_model, tokenizer = tokenizer,
                                   dataset = dataset, seq_len = 384, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_652067.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
########### Untrained embedding ##########
##########################################
accelerator = Accelerator()

# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

##### aid1053197 #####

dataset = aid1053197

cv_accs, cv_labs, cv_preds = train(embedding_model = smiles_model, tokenizer = tokenizer,
                                   dataset = dataset, seq_len = 384, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_1053197_untrained.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##### aid652067 #####

dataset = aid652067

accelerator = Accelerator()

# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

cv_accs, cv_labs, cv_preds = train(embedding_model = smiles_model, tokenizer = tokenizer,
                                   dataset = dataset, seq_len = 384, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_652067_untrained.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)


#####################################
### Prepare BERT model - smilesPE ###
#####################################

spe_vob = codecs.open('data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
spe_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")

device = "cuda"
d_model = 90
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 113


accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 30)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/spe_ckpt/")


##########################################
### dataset AID 1053197 ###
##########################################

dataset = aid1053197


cv_accs, cv_labs, cv_preds = trainPE(embedding_model = smiles_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_1053197.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
### dataset AID 652067 ###
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 30)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/spe_ckpt/")

dataset = aid652067

cv_accs, cv_labs, cv_preds = trainPE(embedding_model = smiles_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_652067.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
########### Untrained embedding ##########
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 30)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

##### aid1053197 #####

dataset = aid1053197

cv_accs, cv_labs, cv_preds = trainPE(embedding_model = smiles_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_1053197_untrained.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##### aid652067 #####

dataset = aid652067

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 30)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

cv_accs, cv_labs, cv_preds = trainPE(embedding_model = smiles_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_652067_untrained.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)



###########################################
### Prepare LARGE BERT model - smilesPE ###
###########################################

spe_vob = codecs.open('data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
spe_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")

device = "cuda"
d_model = 510
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 113


accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/spe_large_ckpt/")


##########################################
### dataset AID 1053197 ###
##########################################

dataset = aid1053197


cv_accs, cv_labs, cv_preds = trainPE_large(embedding_model = smiles_model, 
                                           tokenizer = spe_tokenizer, spe = spe,
                                           dataset = dataset, seq_len = 113, n_tokens = 113,
                                           batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_large113_1053197.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
### dataset AID 652067 ###
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/spe_large_ckpt/")

dataset = aid652067

cv_accs, cv_labs, cv_preds = trainPE_large(embedding_model = smiles_model, 
                                           tokenizer = spe_tokenizer, spe = spe,
                                           dataset = dataset, seq_len = 113, n_tokens = 113,
                                           batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiment_results/res_bert_spe_large113_652067.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
########### Untrained embedding ##########
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

##### aid1053197 #####

dataset = aid1053197

cv_accs, cv_labs, cv_preds = trainPE_large(embedding_model = smiles_model, 
                                           tokenizer = spe_tokenizer, spe = spe,
                                           dataset = dataset, seq_len = 113, n_tokens = 113,
                                           batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_large113_1053197_untrained.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##### aid652067 #####

dataset = aid652067

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

cv_accs, cv_labs, cv_preds = trainPE_large(embedding_model = smiles_model, 
                                           tokenizer = spe_tokenizer, spe = spe,
                                           dataset = dataset, seq_len = 113, n_tokens = 113,
                                           batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_bert_spe_large113_652067_untrained.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
############### Transformer ##############
##########################################

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)
checkpoint = torch.load("checkpoints/transformer_ckpt.ckpt")

transf_model.load_state_dict(checkpoint['state_dict'])

##### aid1053197 #####

dataset = aid1053197

cv_accs, cv_labs, cv_preds = trainT(embedding_model = transf_model,
                                   dataset = dataset, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_transformer_1053197.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##### aid652067 #####

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)
checkpoint = torch.load("checkpoints/transformer_ckpt.ckpt")

transf_model.load_state_dict(checkpoint['state_dict'])

dataset = aid652067


cv_accs, cv_labs, cv_preds = trainT(embedding_model = transf_model,
                                   dataset = dataset, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_transformer_652067.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)


##########################################
########### Untrained embedding ##########
##########################################

TRANSFORMER_DEVICE = "cuda"
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)

##### aid1053197 #####

dataset = aid1053197

cv_accs, cv_labs, cv_preds = trainT(embedding_model = transf_model,
                                   dataset = dataset, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_transformer_1053197_untrained.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##### aid652067 #####

dataset = aid652067

cv_accs, cv_labs, cv_preds = trainT(embedding_model = transf_model,
                                   dataset = dataset, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_transformer_652067_untrained.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##########################################
############### Fine-tuned Transformer ##############
##########################################

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = TransformerBert(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
#transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)
checkpoint = torch.load("checkpoints/finetuned_ckp.pth")

transf_model.load_state_dict(checkpoint['state_dict'])

##### aid1053197 #####

dataset = aid1053197

cv_accs, cv_labs, cv_preds = trainF(embedding_model = transf_model,
                                   dataset = dataset, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_finetuned_transformer_1053197.npz", accs=comp_accs, labs=comp_labs, preds=comp_preds)

##### aid652067 #####

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = TransformerBert(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
#transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)
checkpoint = torch.load("checkpoints/finetuned_ckp.pth")

transf_model.load_state_dict(checkpoint['state_dict'])

dataset = aid652067

cv_accs, cv_labs, cv_preds = trainF(embedding_model = transf_model,
                                   dataset = dataset, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_finetuned_transformer_652067.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)

### JOINT MODELS ###

#####################################
### Prepare BERT model - smilesPE ###
#####################################

spe_vob = codecs.open('data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
spe_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")

device = "cuda"
d_model = 510
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 113


accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/spe_large_ckpt/")

##########################################
############### Transformer ##############
##########################################

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)
checkpoint = torch.load("checkpoints/transformer_ckpt.ckpt")
    
transf_model.load_state_dict(checkpoint['state_dict'])


##########################################
### dataset AID 1053197 ###
##########################################

dataset = aid1053197


cv_accs, cv_labs, cv_preds = trainJ(spe_model = smiles_model, transformer = transf_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_joint_1053197.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)

##########################################
### dataset AID 652067 ###
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/spe_large_ckpt/")

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)
checkpoint = torch.load("checkpoints/transformer_ckpt.ckpt")
    
transf_model.load_state_dict(checkpoint['state_dict'])



dataset = aid652067

cv_accs, cv_labs, cv_preds = trainJ(spe_model = smiles_model, transformer = transf_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_joint_652067.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)

### untrained ####

#####################################
### Prepare BERT model - smilesPE ###
#####################################

spe_vob = codecs.open('data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
spe_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")

device = "cuda"
d_model = 510
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 113


accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)

##########################################
############### Transformer ##############
##########################################

TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)


##########################################
### dataset AID 1053197 ###
##########################################

dataset = aid1053197


cv_accs, cv_labs, cv_preds = trainJ(spe_model = smiles_model, transformer = transf_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_joint_1053197_untrained.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)

##########################################
### dataset AID 652067 ###
##########################################

accelerator = Accelerator()

# Model configuration
vocab_size = len(spe_tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, n_tokens = 113)
smiles_model.to(device)

# accelerate
smiles_model = accelerator.prepare(smiles_model)


TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transf_model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
transf_model = torch.nn.DataParallel(transf_model)
transf_model = transf_model.to(TRANSFORMER_DEVICE)


dataset = aid652067

cv_accs, cv_labs, cv_preds = trainJ(spe_model = smiles_model, transformer = transf_model, tokenizer = spe_tokenizer, spe = spe,
                                   dataset = dataset, seq_len = 113, batch_size = 64, lr=0.001, device = "cuda")

comp_accs, comp_labs, comp_preds = [], [], []

comp_accs.append(np.array(cv_accs))
comp_labs.append(np.array(cv_labs))
comp_preds.append(np.array(cv_preds))

comp_accs = np.array(comp_accs)
comp_labs = np.array(comp_labs)
comp_preds = np.array(comp_preds)

np.savez("Experiments_results/res_joint_652067_untrained.npz", 
         accs=comp_accs, labs=comp_labs, preds=comp_preds)


