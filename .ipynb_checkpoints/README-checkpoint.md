# Smile-to-Bert

Molecular property prediction is crucial for drug discovery. Over the years, deep learning models have been widely used for these tasks; however, large datasets are often needed to achieve strong performances. Pre-training models on vast unlabeled data has emerged as a method to extract contextualized embeddings that boost performance on smaller datasets. The Simplified Molecular Input Line Entry System (SMILES) encodes molecular structures as strings, making them suitable for natural language processing. Transformers, known for capturing long-range dependencies, are well suited for processing SMILES. One such transformer-based architecture is Bidirectional Encoder Representations from Transformers (BERT), which only uses the encoder part of the Transformer and performs classification and regression tasks. Pre-trained transformer-based architectures using SMILES have significantly improved predictions on smaller datasets. Public data repositories such as PubChem, which provide SMILES, among other data, are essential for pre-training these models. 
SMILES embeddings that combine chemical structure and physicochemical property information could further improve performance on tasks such as Absorption, Distribution, Metabolism, Excretion, and Toxicity prediction. To this end, we introduce Smile-to-Bert, a pre-trained BERT architecture designed to predict 113 RDKit-computed molecular descriptors from PubChem SMILES. This model generates embeddings that integrate both molecular structure and physicochemical properties. We evaluate Smile-to-Bert on 22 datasets from the Therapeutics Data Commons and compare its performance with that of the 2-encoder model and a Transformer model. Smile-to-Bert achieves the best result on one dataset, while the combination of Smile-to-Bert with the other models leads to improved performance on 8 datasets. Additionally, the state-of-the-art Transformer is applied to Absorption, Distribution, Metabolism, Excretion, and Toxicity prediction for the first time, achieving the best performance on the Therapeutics Data Commons leaderboard of one dataset.

This repository contains the code and data used for training the models and reproducing the results presented in the paper.

The paper is available as a preprint at https://www.biorxiv.org/content/10.1101/2024.10.31.621293v1.

## Installation

1. Install Python3 if required.
2. Git clone this repository:
```
git clone https://github.com/m-baralt/smile-to-bert.git
```
3. (Optional but recommended) Create a conda environment and activate it:
```
conda create --name smiletobert python=3.11.5
conda activate smiletobert
```
5. Install the required python libraries using:
```
python3 -m pip install -r requirements.txt
```

## Pre-trained model

In order to use the pre-trained model, the weights need to be downloaded using the `download_ckpt.sh` file. To download data using .sh files, the following commands need to be run:
```
chmod +x download_ckpt.sh
./download_ckpt.sh
```
This should create a directory named ckpts with all the weights from the pre-trained model. 
To load the weights of Smile-to-Bert, the following code needs to be executed:

```
import os
import sys
from transformers import BertTokenizer
from accelerate import Accelerator
sys.path.append(os.getcwd())
from Model.BERT import BERT, SMILESLM

smiles_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")
device = "cuda"
d_model = 512
n_layers = 4
heads = 8
dropout = 0.1
seq_length = 100

accelerator = Accelerator()
# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads,
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model, output = 113)
smiles_model.to(device)

smiles_model = accelerator.prepare(smiles_model)
accelerator.load_state(input_dir = "ckpts/smiletobert_ckpt/")
```

To use this model, SMILES strings need to be converted to token sequences using the SMILES Pair Encoding tokenizer. Strings can be converted to tokens sequences using the following lines of code:

```
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer

spe_vob = codecs.open('data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
spe_tokenizer = BertTokenizer("data/spe_tokenizer/vocab_spe.txt")

tokens = tokenizer.encode(spe.tokenize(smiles).split(' '))[0:-1]
padding = [tokenizer.encode('[PAD]')[1] for _ in range(seq_length - len(tokens))]
tokens.extend(padding)
tokens = torch.tensor(tokens)
```

## Embeddings

The ```embedding_viz.py``` file allows the generation and visualization of SMILES embeddings.

```
python3 embedding_viz.py
```

## Training model

```Training_spe.py``` file can be used to pre-train Smile-to-Bert from scratch.
The training and validation data can be downloaded executing the ```download_data.sh``` file. Please consider that the uncompressed data occupies 7.1 GB.

