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
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
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

random.seed(2606)

class SMILESDataset(Dataset):
    def __init__(self, files_index, properties_tensor, median, Q1, Q3, num_samples, seq_len):
        self.files_index = files_index[0:num_samples]
        self.properties_tensor = properties_tensor[0:num_samples]
        self.median = median
        self.Q1 = Q1
        self.Q3 = Q3
        self.seq_len = seq_len
        self.len_compounds = len(self.properties_tensor)

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):    
        file = f'data_atomlevel/data_tensors/tensor_files/{os.path.basename(str(self.files_index[item][0], encoding="utf-8"))}'
        idx = int(str(self.files_index[item][1], encoding='utf-8'))
        smiles_tensor = np.load(file)
        smiles = torch.tensor(smiles_tensor[idx][0:self.seq_len]).type(torch.LongTensor)
        #smiles = smiles[0:self.seq_len]
        properties = self.properties_tensor[item]
        properties = ((properties-self.median)/(self.Q3-self.Q1))*100

        return smiles, properties

class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len, device = 'cuda'):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe = pe.to(device)
        pe.require_grad = False

        for pos in range(max_len):
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
    """

    def __init__(self, vocab_size, embed_size, seq_len, dropout=0.1, device = 'cuda'):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        print('Before embedding')
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0).to(device)
        print('After embedding')
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len, device = device)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
    
class MultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)    

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)

        weights = self.dropout(weights)
        
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        d_model,
        heads, 
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        feed_forward_hidden = d_model*4
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    
class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model, seq_len, n_layers, heads, dropout=0.1, device = 'cuda'):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads


        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, seq_len = seq_len, embed_size=d_model, device = device)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
class PropertiesPrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden, 7)
        #self.linear5 = torch.nn.Linear(3500, 7)
        #self.softmax = torch.nn.LogSoftmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.linear1(x) #self.softmax(self.linear(x))

class SMILESLM(torch.nn.Module):
    
    def __init__(self, bert_model, n_tokens = 83, dropout = 0.1):
        
        super().__init__()
        self.bert = bert_model
        self.propsPrediction = PropertiesPrediction(self.bert.d_model*n_tokens)
        self.n_tokens = n_tokens

    def forward(self, smiles):
        smiles = self.bert(smiles)
        #smiles2 = torch.flatten(smiles[:,0:83])
        #smiles2 = torch.stack([row for row in smiles[:,0.83]])
        smiles2 = smiles[:, :self.n_tokens, :].reshape(smiles.size(0), -1)
        return self.propsPrediction(smiles2), smiles2

