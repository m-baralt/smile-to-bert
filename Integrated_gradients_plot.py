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

parser = argparse.ArgumentParser()

parser.add_argument("SMILES", type=str, help="SMILES string to process.")
parser.add_argument("--print_properties", help="Print predicted properties.", action="store_true")
parser.add_argument("--cpu", help="Use CPUs.", action="store_true")
parser.add_argument("--figure_path", type=str, default="Integrated_gradients.png", help="Path to save the integrated gradients figure generated.")

args = parser.parse_args()



### Data

# data
tokenizer = BertTokenizer("data/atomlevel_tokenizer/vocab.txt")

iqr_properties = torch.load("data/iqr_properties.pt")
Q1 = iqr_properties[0]
median = iqr_properties[1]
Q3 = iqr_properties[2]

d_model = 90
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 384

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

# accelerate
smiles_model = accelerator.prepare(smiles_model)

accelerator.load_state(input_dir = "checkpoints/atomlevel_ckp/")

def predict(inputs):
    smiles_model.eval()
    properties, embed = smiles_model(inputs)
    return properties

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

def construct_input_ref_pair(smiles, ref_token_id, cls_token_id):
    basictokenizer = BasicSmilesTokenizer(SMI_REGEX_PATTERN)
    tokens = tokenizer.encode(basictokenizer.tokenize(smiles))[0:-1]
    padding = [tokenizer.encode('[PAD]')[1] for _ in range(384 - len(tokens))]
    tokens.extend(padding)
    #pred = smiles_model(tokens)
    tokens = torch.tensor(tokens)
    tokens = tokens.view(1,384).to('cuda')


    # construct input token ids
    input_ids = tokens

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * 383

    return input_ids, torch.tensor([ref_input_ids], device=device), 384

def construct_input_ref_token_type_pair(input_ids):
    seq_len = input_ids.size(1)
    ref_token_type_ids = torch.zeros(seq_len, dtype = torch.int32, device=device)# * -1
    return ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_whole_bert_embeddings(input_ids, ref_input_ids):
    
    input_embeddings = smiles_model.bert.embedding(input_ids)
    ref_input_embeddings = smiles_model.bert.embedding(ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

input_ids, ref_input_ids, sep_id = construct_input_ref_pair(args.SMILES, ref_token_id, cls_token_id)
ref_token_type_ids = construct_input_ref_token_type_pair(input_ids)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)
indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

predicted_props = predict(input_ids)
props_names = ["count of H-bond acceptor", "count of H-bond donor", "count of rotatable bond", "exact mass", "TPSA","count of heavy atom", "log-P"]

if (args.print_properties):
    for i,val in enumerate(props_names):
        pred = ((predicted_props[0][i]/100)*(Q3[i]-Q1[i]))+median[i]
        print(f"The predicted {val} is {pred:.2f}")

lig = LayerIntegratedGradients(predict, smiles_model.bert.embedding)

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)
all_tokens = all_tokens[1:]

new_tokens = []
for t in all_tokens:
    if not '[PAD]' in t:
        new_tokens.append(t)

scores_allprops = torch.zeros((len(props_names),len(new_tokens)))

attributions_all_outputs = []
deltas_all_outputs = []
for output_idx in range(len(props_names)):  # Assuming we have 3 outputs
    predictions = predict(input_ids)
    predictions = predictions[0][output_idx]
    attributions, delta = lig.attribute(inputs=input_ids,
                                       target=output_idx,
                                       return_convergence_delta=True)
    scores = attributions.sum(dim = -1)
    scores = (scores-scores.mean())/scores.norm()
    log_vector = []
    for t in all_tokens:
        if '[PAD]' in t:
            log_vector.append(False)
        else:
            log_vector.append(True)

    scores = scores.squeeze()
    scores = scores[1:]
    scores = scores[log_vector]

    scores_allprops[output_idx,:] = scores


fig, ax = plt.subplots(figsize=(15,5))
xticklabels=new_tokens
yticklabels=["H-bond acceptor", "H-bond donor", "Rotatable bond", "Exact mass", "TPSA","Heavy atom", "Log-P"]
ax = sns.heatmap(np.array(scores_allprops), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2, cmap = 'plasma')
plt.show()

plt.savefig(args.figure_path)



