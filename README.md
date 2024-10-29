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

In order to use the pre-trained models, the weights need to be downloaded using the `download_ckpt.sh` file. To download data using .sh files, the following commands neet to be run:
```
chmod +x download_ckpt.sh
./download_ckpt.sh
```
This should create a directory named checkpoints with all the weights from the pre-trained models. 
To load the weights of the atom-level Smile-to-Bert, the following code needs to be executed:

```
import os
import sys
from transformers import BertTokenizer
from accelerate import Accelerator
sys.path.append(os.getcwd())
from Model.BERT import BERT, SMILESLM

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

To use this model, SMILES strings need to be converted to token sequences using the atom-level tokenizer from deepchem. To do so, the following code is required:

```
import re
import torch

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

smiles = "C(C1C(C(C(C(O1)O)O)O)O)O"
basictokenizer = BasicSmilesTokenizer(SMI_REGEX_PATTERN)
tokens = tokenizer.encode(basictokenizer.tokenize(smiles))[0:-1]
padding = [tokenizer.encode('[PAD]')[1] for _ in range(seq_length - len(tokens))]
tokens.extend(padding)
tokens = torch.tensor(tokens)
```

For the SmilesPE tokenizer, the SMILES strings need to be converted to a tokens sequence using the following lines of code:

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

## Properties prediction and integrated gradients

The ```Integrated_gradients_plot.py``` file allows an easy prediction of physicochemical properties from one SMILES string, and it generates a plot from the integrated gradients algorithm from Captum for the atom-level Smile-to-Bert. Please, make sure that the directory specified in ```figure_path``` exists:

```
mkdir results
python3 Integrated_gradients_plot.py "C(C1C(C(C(C(O1)O)O)O)O)O" --print_properties --figure_path=results/your_smiles_gradients.png
```

## Embeddings

The ```embedding_viz.py``` file allows the generation of SMILES embeddings from a txt file with SMILES strings. Additionally, the PCAs of the embeddings are visualised. An example of a file with SMILES strings can be downloaded using ```download_smiles.sh```. This should create a directory named smiles with the smiles_string.txt file.

```
python3 embedding_viz.py --data_path=/smiles/smiles_string.txt --figure_path=results/
```

## Training model

To train the atom-level and the SmilesPE Smile-to-Bert, the files ```Training_atomlevel.py``` and ```Training_spe.py``` can be used respectively. 
The training and validation data can be downloaded executing the ```download_atomlevel_data.sh``` and/or ```download_spe_data.sh``` files.

## Binding affinity prediction

Finally, the file ```BERT_transfer_results.py``` can be used to run all the experiments related to binding affinity prediction. The datasets can be downloaded using the ```download_experiments.sh```file. 



