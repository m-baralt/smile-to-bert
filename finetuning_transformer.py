import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
import copy
from torch.utils.data import random_split
import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from transformers import BertTokenizer
from torch.optim import Adam
import wandb
import gc

os.makedirs("Training_results")

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_entity", type=str, help="Wandb username.")
args = parser.parse_args()



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

PRINTABLE_ASCII_CHARS = 95

_extra_chars = ["seq_start", "seq_end", "pad"]
EXTRA_CHARS = {key: chr(PRINTABLE_ASCII_CHARS + i) for i, key in enumerate(_extra_chars)}
ALPHABET_SIZE = PRINTABLE_ASCII_CHARS + len(EXTRA_CHARS)

class OneHotEmbedding(nn.Module):
    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.embedding = nn.Embedding.from_pretrained(torch.eye(alphabet_size))
    def forward(self, x):
        return self.embed(x)
    
class Embedding(nn.Module):
    def __init__(self, alphabet_size, d_model):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.d_model = d_model
        self.embed = nn.Embedding(alphabet_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 6000, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        #print(x.mean(), x)
        x = self.dropout(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        #print(x.mean(), x)
        return x

    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, alphabet_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding(alphabet_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, alphabet_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding(alphabet_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, alphabet_size, d_model, N, heads=8, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(alphabet_size, d_model, N, heads, dropout)
        self.decoder = Decoder(alphabet_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, alphabet_size)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

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


class TransformerBert(nn.Module):
    def __init__(self, alphabet_size, d_model, N, heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(alphabet_size, d_model, N, heads, dropout)
        self.propsPrediction = PropertiesPrediction(self.d_model*256)
        
        #self.decoder = Decoder(alphabet_size, d_model, N, heads, dropout)
        #self.out = nn.Linear(d_model, alphabet_size)
    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        e_outputs = e_outputs.view(-1, self.d_model*256)
        output = self.propsPrediction(e_outputs)
        
        #print("DECODER")
        #d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        #output = self.out(d_output)
        return output

    
def nopeak_mask(size, device):
    np_mask = torch.triu(torch.ones((size, size), dtype=torch.uint8), diagonal=1).unsqueeze(0)
    
    np_mask = np_mask == 0
    np_mask = np_mask.to(device)
    return np_mask

def create_masks(src, trg=None, pad_idx=ord(EXTRA_CHARS['pad']), device=None):
    src_mask = (src != pad_idx).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, device)
        np_mask.to(device)
        trg_mask = trg_mask & np_mask
        return src_mask, trg_mask
    return src_mask

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
model = torch.nn.DataParallel(model)
model = model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
#sched = CosineWithRestarts(optimizer, T_max=len(dataloader))

checkpoint = torch.load("checkpoints/pretrained.ckpt")
model.load_state_dict(checkpoint['state_dict'])

new_model = TransformerBert(ALPHABET_SIZE, 512, 6)
#if torch.cuda.is_available() and not args.cpu:
new_model = torch.nn.DataParallel(new_model)
new_model = new_model.to("cuda")
new_model_state_dict = new_model.state_dict()

excluded_layers = ['module.decoder', 'module.out']
for name, param in checkpoint['state_dict'].items():
    if name in new_model_state_dict and not any(layer in name for layer in excluded_layers):
        new_model_state_dict[name].copy_(param)


new_model.load_state_dict(new_model_state_dict)



### data
# data
smiles_tokenizer = BertTokenizer("data/atomlevel_tokenizer/vocab.txt")
# Load data
properties_tensor = torch.load('data_atomlevel/filt_props_tensor.pt').numpy()
ifile = np.load("data_atomlevel/data_tensors/file_index.npy")

num_samples = len(properties_tensor)
properties_tensor = properties_tensor[0:num_samples]

Q1 = []
Q3 = []
median = []
for j in range(properties_tensor.shape[1]):
    Q1.append(np.quantile(properties_tensor[:,j], 0.25))
    Q3.append(np.quantile(properties_tensor[:,j], 0.75))
    median.append(np.median(properties_tensor[:,j]))

Q1 = torch.tensor(Q1)
Q3 = torch.tensor(Q3)
median = torch.tensor(median)


# Hbond acceptor, Hbond donor, Rotatable bond, Exact mass, TPSA, Heavy atom count, Log-P
properties_tensor = torch.load('data_atomlevel/filt_props_tensor.pt')


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

MAX_LEN = 256
MODEL_DIM = 512
N_LAYERS = 6

class SMILESDataset(Dataset):
    def __init__(self, files_index, tokenizer, properties_tensor, median, Q1, Q3, num_samples, seq_len):
        self.files_index = files_index[0:num_samples]
        self.properties_tensor = properties_tensor[0:num_samples]
        self.median = median
        self.Q1 = Q1
        self.Q3 = Q3
        self.seq_len = seq_len
        self.len_compounds = len(self.properties_tensor)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len_compounds

    def __getitem__(self, item):
        file = f'data_atomlevel/data_tensors/tensor_files/{os.path.basename(str(self.files_index[item][0], encoding="utf-8"))}'
        idx = int(str(self.files_index[item][1], encoding='utf-8'))
        smiles_tensor = np.load(file)
        smiles_ids = torch.tensor(smiles_tensor[idx][0:self.seq_len]).type(torch.LongTensor)
        smiles_ids = smiles_ids[smiles_ids != 0]
        tokens = self.tokenizer.convert_ids_to_tokens(smiles_ids)
        smiles = "".join(tokens[1:])
        encoded = encode_string_np(smiles, start_char=EXTRA_CHARS['seq_start'], pad_char=EXTRA_CHARS['pad'])
        smiles = torch.tensor(encoded, dtype = torch.int)
        #smiles = smiles[0:self.seq_len]
        properties = self.properties_tensor[item]
        properties = ((properties-self.median)/(self.Q3-self.Q1))*100

        return smiles, properties


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_current_steps = 0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            

def loss_per_prop(pred, Y, median, Q1, Q3, loss_fn, num_properties):
    one_batch_loss = []
    for i in range(num_properties):
        one_prop_pred = pred[:,i].clone().detach()
        one_prop_y = Y[:,i].clone().detach()
        
        one_prop_pred = ((one_prop_pred/100)*(Q3[i]-Q1[i]))+median[i]
        one_prop_y = ((one_prop_y/100)*(Q3[i]-Q1[i]))+median[i]
        loss_prop = loss_fn(one_prop_pred, one_prop_y)
        one_batch_loss.append(loss_prop)
    return one_batch_loss


def prepare_data(tokenizer, num_samples, files_index, properties_tensor,  
                 batch_size, median, Q1, Q3, seq_len, train_perc, num_workers, seed = 10):
    
    smiles_dataset = SMILESDataset(tokenizer = tokenizer,
                                   files_index = files_index,
                                   properties_tensor = properties_tensor, 
                                   median = median, Q1 = Q1, Q3 = Q3, 
                                   num_samples = num_samples, seq_len = seq_len)
    
    generator1 = torch.Generator().manual_seed(seed)
    train_size = int(len(smiles_dataset) * train_perc)
    trainset, testset = random_split(
        dataset = smiles_dataset, 
        lengths = [train_size, len(smiles_dataset) - train_size], 
        generator = generator1
    )
    
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dataloader = DataLoader(dataset = trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers, 
                                  worker_init_fn=seed_worker, generator=g)
    
    #eval_dataloader = DataLoader(dataset = testset, shuffle=False, batch_size=batch_size, num_workers=4)
    
    return train_dataloader, testset, len(trainset)

def validation(model, testset, device, median, Q1, Q3, batch_size, num_workers, num_properties):#, tokenizer, num_samples, props_array, batch_size, smiles_list, mean, sd):
    
    #train_loader, test_loader = prepare_data(num_samples = num_samples, properties_array = props_array, 
                                 #batch_size = batch_size, smiles_list = smiles_list, tokenizer = tokenizer, 
                                 #mean = mean, sd = sd, seq_len = 384, seed = 2)

    test_loader = DataLoader(dataset = testset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    batch_loss = []
    batch_loss_per_prop = []
    
    #  defining model state
    model.eval()
    
    loss_fn = nn.L1Loss().to(device)
    
    print('validating...')
    #  preventing gradient calculations since we will not be optimizing
    with torch.no_grad():
        for batch, (X, Y) in tqdm.tqdm(enumerate(test_loader)):
            X = X.to(device)
            Y = Y.to(device)
            mask = create_masks(X)
            pred = model(X, mask)
            loss = loss_fn(pred, Y)
            loss_list = loss_per_prop(pred = pred, Y = Y, median = median, Q1 = Q1, Q3 = Q3, loss_fn = loss_fn, num_properties = num_properties)
            batch_loss.append(loss.item())
            batch_loss_per_prop.append(loss_list)
            
        # computing the mean of the batch losses per property    
        d = {}
        for elem in batch_loss_per_prop:
            for i in range(len(elem)):
                if i in d:
                    d[i].append(elem[i])
                else:
                    d[i] = [elem[i]]

        mean_list = []
        for k,v in d.items():
            mean_list.append(sum(v) / len(v))
            

        return np.mean(batch_loss), mean_list


def training_loop(
    model,
    tokenizer,
    files_index,
    properties_tensor,
    median, 
    Q1, 
    Q3, 
    d_model,
    n_layers,
    heads,
    num_samples,
    batch_size,
    dropout = 0.1,
    seq_length = 384,
    num_properties = 7,
    lr = 1e-4,
    betas = (0.9, 0.999),
    weight_decay = 0.01,
    warmup_steps=10000,
    start_epoch = 0,
    stop_epoch = 100,
    checkpoint_path = None,
    num_workers = 15,
    device = 'cuda'
):
    
    
    # data preparation
    train_loader, testset, train_size = prepare_data(tokenizer = tokenizer, 
                                                     num_samples = num_samples, 
                                                     files_index = files_index,
                                                     properties_tensor = properties_tensor, 
                                                     batch_size = batch_size, median = median, Q1 = Q1, Q3 = Q3, 
                                                     seq_len = seq_length, train_perc = 0.95, 
                                                     num_workers = num_workers, seed = 2)

    run = wandb.init(
        # Set the project where this run will be logged
        project="BERT-smiles-training",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": stop_epoch,
    }, entity = args.wandb_entity)


    # optimization
    loss_fn = nn.L1Loss().to(device)
    optimizer = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    optim_schedule = ScheduledOptim(optimizer = optimizer, d_model = d_model, n_warmup_steps=warmup_steps, 
                                    n_current_steps = 0)
    
    
    # setting model state
    model.train()
    
    # losses lists
    train_loss = []
    train_loss_per_prop = []
    val_loss_list = []
    
    # load accelerator state dict in case of checkpoint
    
    print("Training size: ", train_size)
    print("Validation size: ", len(testset))
    custom_batch = -1
    for t in range(0, stop_epoch):
        
        print(f"Epoch {t}\n-------------------------------")
        
        # losses list
        batch_loss = 0
        batch_loss_per_prop = [0]*num_properties

        batch_loss_per_prop1000 = [0]*num_properties
        
        for batch, (X, Y) in tqdm.tqdm(enumerate(train_loader)):
            custom_batch = custom_batch+1
            X = X.to(device)
            Y = Y.to(device)
            
            #forward pass
            mask = create_masks(X)
            pred = model(X, mask)
            
            # loss computation
            loss = loss_fn(pred, Y)
            
            # loss per property computation
            loss_list = loss_per_prop(pred = pred, Y = Y, median = median, 
                                      Q1 = Q1, Q3 = Q3, loss_fn = loss_fn, 
                                      num_properties = num_properties)

            #import pdb; pdb.set_trace()


            loss.backward()
            
            # optimizer step and scheduler update
            optim_schedule.step_and_update_lr()
            optim_schedule.zero_grad()
            # batch losses list appending
            batch_loss = batch_loss+loss.item()
            
            #batch_loss.append(loss.item())
            
            batch_loss_per_prop_temp = []
            for i in range(len(loss_list)):
                batch_loss_per_prop_temp.append(loss_list[i]+batch_loss_per_prop[i])
                
            batch_loss_per_prop = batch_loss_per_prop_temp

            ## reset mean
            batch_loss_per_prop1000_temp = []
            for i in range(len(loss_list)):
                batch_loss_per_prop1000_temp.append(loss_list[i]+batch_loss_per_prop1000[i])
                
            batch_loss_per_prop1000 = batch_loss_per_prop1000_temp


            if ((batch==0) | (batch%1000==999)):
                mean_loss = batch_loss/(batch+1)
                batch_loss_per_prop_mean = [val/(batch+1) for val in batch_loss_per_prop]
                    
                batch_loss_per_prop1000_mean = [val/1000 for val in batch_loss_per_prop1000]
                if batch==0:
                    batch_loss_per_prop1000_mean = batch_loss_per_prop1000
                # Hbond acceptor, Hbond donor, Rotatable bond, Exact mass, TPSA, Heavy atom count, Log-P
                wandb.log({"epoch": t, 
                                 "batch": batch,
                                 "Loss": mean_loss,#loss.item(), 
                                 "Hbond Acceptor": batch_loss_per_prop_mean[0], #loss_list[1], 
                                 "Hbond Donor":batch_loss_per_prop_mean[1], #loss_list[2], 
                                 "Rotatable Bond": batch_loss_per_prop_mean[2], #loss_list[3], 
                                 "Exact Mass": batch_loss_per_prop_mean[3], #loss_list[4],
                                 "TPSA": batch_loss_per_prop_mean[4], #loss_list[6],
                                 "Heavy Atom Count": batch_loss_per_prop_mean[5], #loss_list[8], 
                                 "Log-P": batch_loss_per_prop_mean[6],
                                 "Reset Hbond Acceptor": batch_loss_per_prop1000_mean[0], #loss_list[1], 
                                 "Reset Hbond Donor":batch_loss_per_prop1000_mean[1], #loss_list[2], 
                                 "Reset Rotatable Bond": batch_loss_per_prop1000_mean[2], #loss_list[3], 
                                 "Reset Exact Mass": batch_loss_per_prop1000_mean[3], #loss_list[4],
                                 "Reset TPSA": batch_loss_per_prop1000_mean[4], #loss_list[6],
                                 "Reset Heavy Atom Count": batch_loss_per_prop1000_mean[5], #loss_list[8], 
                                 "Reset Log-P": batch_loss_per_prop1000_mean[6]}) #loss_list[16]})   

                batch_loss_per_prop1000 = [0]*num_properties
                gc.collect()
        
        # train losses appending
        
        
        batch_loss_mean = batch_loss/(batch+1)
        train_loss.append(batch_loss_mean)
        
        batch_loss_per_prop_mean = [val/(batch+1) for val in batch_loss_per_prop]
        train_loss_per_prop.append(batch_loss_per_prop_mean)

        torch.save({'state_dict': model.module.state_dict()}, checkpoint_path+"checkpoint_"+str(t)+".pth")

        torch.save(obj = {
            'epoch': t,
            'scheduler_current_step': optim_schedule.n_current_steps,
            'loss': train_loss[-1]
            }, f = checkpoint_path+"meta_model_"+str(t)+".tar")

        
        # Mean computation of the losses per property
        #d = {}
        #for elem in batch_loss_per_prop:
         #   for i in range(len(elem)):
        #     if i in d:
         #           d[i].append(elem[i])
          #      else:
           #         d[i] = [elem[i]]

        #mean_list = []
        #for k,v in d.items():
         #   mean_list.append(sum(v) / len(v))
        
        #train_loss_per_prop.append(mean_list)

        # validation for each epoch
        val_loss, val_loss_per_prop = validation(model = model, 
                                                 testset = testset, 
                                                 median = median,
                                                 Q1 = Q1, Q3 = Q3, 
                                                 batch_size = batch_size, 
                                                 device = device, 
                                                 num_workers = num_workers, 
                                                 num_properties = num_properties)
        
        val_loss_list.append([val_loss, val_loss_per_prop])

        # Hbond acceptor, Hbond donor, Rotatable bond, Exact mass, TPSA, Heavy atom count, Log-P
        wandb.log({"epoch": t, 
                         "batch": custom_batch,
                         "Epoch Loss": train_loss[-1],
                         "Epoch validation loss": val_loss,
                         "Epoch Hbond Acceptor": batch_loss_per_prop_mean[0], 
                         "Epoch Hbond Donor":batch_loss_per_prop_mean[1], 
                         "Epoch Rotatable Bond": batch_loss_per_prop_mean[2], 
                         "Epoch Exact Mass": batch_loss_per_prop_mean[3], 
                         "Epoch TPSA": batch_loss_per_prop_mean[4],
                         "Epoch Heavy Atom Count": batch_loss_per_prop_mean[5], 
                         "Epoch Log-P": batch_loss_per_prop_mean[6],
                         "Validation Hbond Acceptor": val_loss_per_prop[0], 
                         "Validation Hbond Donor":val_loss_per_prop[1], 
                         "Validation Rotatable Bond": val_loss_per_prop[2], 
                         "Validation Exact Mass": val_loss_per_prop[3], 
                         "Validation TPSA": val_loss_per_prop[4],
                         "Validation Heavy Atom Count": val_loss_per_prop[5], 
                         "Validation Log-P": val_loss_per_prop[6]})
        
        #output_batch_list.append([batch_loss, batch_loss_per_prop])
        


        print(f"Loss: {train_loss[-1]}")
        print(f"Loss per property: {batch_loss_per_prop_mean}")

        
    output_list = [train_loss, val_loss_list]
    with open(checkpoint_path+"results_list.pkl", 'wb') as f:
        pickle.dump(output_list, f)
    print("Done!")
                

stop_epoch = 100
if __name__ == "__main__":
    training_loop(model = new_model, tokenizer = smiles_tokenizer, files_index = ifile,
                  properties_tensor = properties_tensor, median = median, Q1 = Q1, Q3 = Q3, 
                  start_epoch = 0, stop_epoch = stop_epoch, d_model = 512, num_samples = num_samples,
                  n_layers = N_LAYERS, heads = 8, dropout = 0.1, seq_length = 256, num_properties = 7, num_workers = 10,
                  batch_size = 64, lr = 1e-4, betas = (0.9, 0.999), weight_decay = 0.01, warmup_steps=10000,
                  checkpoint_path = "Training_results/",
                  device = 'cuda')
