#from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from Model.Transformer import EXTRA_CHARS

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

class FC(nn.Module):
    def __init__(self, input_features=90*83, activation=F.relu):
        super(FC, self).__init__()
        self.act = activation
        self.input_size = input_features
        
        self.dim1 = 500
        self.dim2 = 100
        
        self.fc1 = nn.Linear(self.input_size, self.dim1)
        self.fc2 = nn.Linear(self.dim1, self.dim1)
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(self.dim1, self.dim2)
        self.fc4 = nn.Linear(self.dim2, 2)
        
    def forward(self, x):
        #x = x.view(-1, self.input_size)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(self.act(x))
        
        x_int = self.fc3(x)
        x = self.fc4(self.act(x_int))
        
        return x, x_int

class FCT(nn.Module):
    def __init__(self, input_features=512*256, activation=F.relu):
        super(FCT, self).__init__()
        self.act = activation
        self.input_size = input_features
        
        self.dim1 = 500
        self.dim2 = 100
        
        self.fc1 = nn.Linear(self.input_size, self.dim1)
        self.fc2 = nn.Linear(self.dim1, self.dim1)
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(self.dim1, self.dim2)
        self.fc4 = nn.Linear(self.dim2, 2)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(self.act(x))
        
        x_int = self.fc3(x)
        x = self.fc4(self.act(x_int))
        
        return x, x_int


class SMILESDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len, idx):
        self.data = data.iloc[idx]
        self.seq_len = seq_len
        self.len_compounds = len(self.data)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):   
        smiles = self.data['CanonicalSMILES'].iloc[item]
        basictokenizer = BasicSmilesTokenizer(SMI_REGEX_PATTERN)
        tokens = self.tokenizer.encode(basictokenizer.tokenize(smiles))[0:-1]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens = torch.tensor(tokens)
        
        label = torch.tensor(self.data['binding'].iloc[item])
    
        return tokens, label

class SMILESDatasetPE(Dataset):
    def __init__(self, data, tokenizer, spe, seq_len, idx):
        self.data = data.iloc[idx]
        self.seq_len = seq_len
        self.len_compounds = len(self.data)
        self.tokenizer = tokenizer
        self.spe = spe

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):   
        smiles = self.data['CanonicalSMILES'].iloc[item]
        tokens = self.tokenizer.encode(self.spe.tokenize(smiles).split(' '))[0:-1]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens = torch.tensor(tokens)  
        label = torch.tensor(self.data['binding'].iloc[item])
    
        return tokens, label


## class obtained from MolecularTransformerEmbeddings github
class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        The maximum number of iterations within the first cycle.
    eta_min : float, optional (default: 0)
        The minimum learning rate.
    last_epoch : int, optional (default: -1)
        The index of the last epoch.
    """

    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min = 0.,
                 last_epoch = -1,
                 factor = 1.):
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = T_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


### Transformer binding

def encode_char(c):
    return ord(c) - 32

def encode_string(string, start_char=chr(0)):
    return torch.tensor([ord(start_char)] + [encode_char(c) for c in string])

def encode_string_np(string, start_char=chr(0), pad_char=chr(0)):
    if len(string) > 255:
        string = string[:255]
        
    arr = np.full((256,), ord(pad_char), dtype=np.float32)
    arr[:len(string)+1] = np.array([ord(start_char)] + [encode_char(c) for c in string])
    return arr


def nopeak_mask(size, device):
    np_mask = torch.triu(torch.ones((size, size), dtype=torch.uint8), diagonal=1).unsqueeze(0)
    
    np_mask = np_mask == 0
    np_mask = np_mask.to(device)
    return np_mask

class SMILESDatasetT(Dataset):
    def __init__(self, data, idx):
        self.data = data.iloc[idx]
        self.len_compounds = len(self.data)

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):   

        smiles = self.data['CanonicalSMILES'].iloc[item]
        encoded = encode_string_np(smiles, start_char=EXTRA_CHARS['seq_start'], pad_char=EXTRA_CHARS['pad'])
        #encoded = encoded[:,:self.seq_len]
        tokens = torch.tensor(encoded, dtype = torch.int)
        
        label = torch.tensor(self.data['binding'].iloc[item])
    
        return tokens, label
        

class SMILESDatasetJ(Dataset):
    def __init__(self, data, idx, tokenizer, spe, seq_len):
        self.data = data.iloc[idx]
        self.len_compounds = len(self.data)

        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.spe = spe

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):   

        smiles = self.data['CanonicalSMILES'].iloc[item]
        encoded = encode_string_np(smiles, start_char=EXTRA_CHARS['seq_start'], pad_char=EXTRA_CHARS['pad'])
        #encoded = encoded[:,:self.seq_len]
        tokens_transf = torch.tensor(encoded, dtype = torch.int)

        tokens_spe = self.tokenizer.encode(self.spe.tokenize(smiles).split(' '))[0:-1]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens_spe))]
        tokens_spe.extend(padding)
        tokens_spe = torch.tensor(tokens_spe)
        
        label = torch.tensor(self.data['binding'].iloc[item])
    
        return tokens_transf, tokens_spe, label






