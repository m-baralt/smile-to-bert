# Smile-to-Bert

Deep learning models have been widely applied in drug discovery and development, with tasks like binding affinity prediction being a common example. However, achieving strong performance with deep learning often requires large datasets, which are not always available in this field. To address this, pre-training models on vast, unlabelled datasets to obtain contextualised embeddings has become a common approach, as these embeddings can boost performance on smaller datasets.

SMILES (Simplified Molecular Input Line Entry System) are text-based representations of molecular structures, making them suitable for natural language processing (NLP) algorithms. Since the order of characters in SMILES does not necessarily respresent physical proximity in the molecule, Transformers are well-suited for processing this data, as they can capture both long-range and short-range dependencies.

In this project, we used a BERT (Bidirectional Encoder Representations from Transformers) architecture to predict physicochemical properties from SMILES and to generate compound embeddings that encode both chemical structure and physicochemical properties. We compared two tokenization methods—atom-level and SmilesPE—and trained and validated models on 99,886,786 compounds for atom-level tokenization and 99,885,676 compounds for SmilesPE tokenization, both sourced from PubChem.

The BERT architectures consisted of 4 encoder layers, 6 attention heads, and an embedding size of 90. The models were trained to predict 7 physicochemical properties: number of H-bond acceptors, number of H-bond donors, number of rotatable bonds, exact mass, topological polar surface area (TPSA), number of heavy atoms, and log-P. The input sequence lengths were fixed at 384 for atom-level tokenization and 113 for SmilesPE.

Additionally, we evaluated the utility of the generated embeddings in a binding affinity prediction task using two datasets and compared their performance to embeddings created by a state-of-the-art Transformer model.

This repository contains the code and data used for training the models and reproducing the results presented in the paper.

## Installation

1. Install Python3 if required.
2. Git clone this repository:
```
git clone https://github.com/m-baralt/smile-to-bert.git
```
3. (Optional) Create a conda environment and activate it
4. Install the required python libraries using:
```
pip3 install -r requirements.txt
```

## Pre-trained model

In order to use the pre-trained models, the weights need to be downloaded using the `download_ckpt.sh` file. This should create a directory named checkpoints with all the weights from the pre-trained models. 
To use the atom-level Smile-to-Bert, the following code needs to be executed:

```
tokenizer = BertTokenizer("data/atomlevel_tokenizer/vocab.txt")
device = "cuda"
d_model = 90
n_layers = 4
heads = 6
dropout = 0.1
seq_length = 384

accelerator = Accelerator()
# Model configuration
vocab_size = len(tokenizer.vocab)
bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads,
                  dropout=dropout, seq_len = seq_length, device = device)
smiles_model = SMILESLM(bert_model = bert_model)
smiles_model.to(device)

smiles_model = accelerator.prepare(smiles_model)
accelerator.load_state(input_dir = "checkpoints/atomlevel_ckp/")
```

## Properties prediction

## Embeddings

## Training model
