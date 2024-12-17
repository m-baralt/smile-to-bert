import torch
import argparse
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
import wandb
import torch.distributed as dist
import datetime
import gc
from transformers import BertTokenizer
import torch.nn.functional as F
sys.path.append(os.getcwd())
from Model.BERT import BERT, SMILESLM

os.makedirs("Training_results")

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_entity", type=str, help="Wandb username.")
args = parser.parse_args()

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

# external functions
# modified scheduler
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


def prepare_data(num_samples, files_index, properties_tensor,  
                 batch_size, median, Q1, Q3, seq_len, train_perc, num_workers, seed = 10):
    
    smiles_dataset = SMILESDataset(files_index = files_index,
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
    
    train_dataloader = DataLoader(dataset = trainset, shuffle=True, 
                                  batch_size=batch_size, num_workers=num_workers, 
                                  worker_init_fn=seed_worker, generator=g)
    
    return train_dataloader, testset, len(trainset)

def validation(model, testset, device, median, Q1, Q3, 
               batch_size, num_workers, num_properties):

    test_loader = DataLoader(dataset = testset, shuffle=False, 
                             batch_size=batch_size, num_workers=num_workers)
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
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss_list = loss_per_prop(pred = pred, Y = Y, 
                                      median = median, Q1 = Q1, 
                                      Q3 = Q3, loss_fn = loss_fn, 
                                      num_properties = num_properties)
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
    
    accelerator = Accelerator(gradient_accumulation_steps=1, 
                              log_with="wandb", 
                              kwargs_handlers = dist.init_process_group(backend='nccl', 
                                                                        init_method='env://', 
                                                                        timeout=datetime.timedelta(seconds=7200)))
    #accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="BERT-smiles-training", 
        config={"learning_rate": lr, "epochs": stop_epoch},
        init_kwargs={"wandb": {"entity": args.wandb_entity}}
    )

    # Model configuration
    vocab_size = len(tokenizer.vocab)
    bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                      dropout=dropout, seq_len = seq_length, device = device)
    smiles_model = SMILESLM(bert_model = bert_model)
    smiles_model.to(device)
    
    # data preparation
    train_loader, testset, train_size = prepare_data(num_samples = num_samples, 
                                                     files_index = files_index,
                                                     properties_tensor = properties_tensor, 
                                                     batch_size = batch_size, median = median, 
                                                     Q1 = Q1, Q3 = Q3, 
                                                     seq_len = seq_length, train_perc = 0.95, 
                                                     num_workers = num_workers, seed = 2)


    # optimization
    loss_fn = nn.L1Loss().to(device)
    optimizer = Adam(smiles_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    optim_schedule = ScheduledOptim(optimizer = optimizer, d_model = d_model, n_warmup_steps=warmup_steps, 
                                    n_current_steps = 0)
    
    # accelerate
    train_loader, smiles_model, optimizer, optim_schedule = accelerator.prepare(train_loader, smiles_model, 
                                                                                optimizer, optim_schedule)
    
    # setting model state
    smiles_model.train()
    
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
            with accelerator.accumulate(smiles_model):
                #forward pass
                pred = smiles_model(X)
                
                # loss computation
                loss = loss_fn(pred, Y)
                
                # loss per property computation
                loss_list = loss_per_prop(pred = pred, Y = Y, median = median, 
                                          Q1 = Q1, Q3 = Q3, loss_fn = loss_fn, 
                                          num_properties = num_properties)
                
                # backward pass
                accelerator.backward(loss)
                # optimizer step and scheduler update
                optim_schedule.step_and_update_lr()
                optim_schedule.zero_grad()
                # batch losses list appending
                batch_loss = batch_loss+loss.item()
                
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
                    accelerator.log({"epoch": t, 
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
        
        accelerator.save_state(output_dir=checkpoint_path+"checkpoint_"+str(t)+"/")

        torch.save(obj = {
            'epoch': t,
            'scheduler_current_step': optim_schedule.n_current_steps,
            'loss': train_loss[-1]
            }, f = checkpoint_path+"meta_model_"+str(t)+".tar")

        
        val_loss, val_loss_per_prop = validation(model = smiles_model, 
                                                 testset = testset, 
                                                 median = median,
                                                 Q1 = Q1, Q3 = Q3, 
                                                 batch_size = batch_size, 
                                                 device = device, 
                                                 num_workers = num_workers, 
                                                 num_properties = num_properties)
        
        val_loss_list.append([val_loss, val_loss_per_prop])

        # Hbond acceptor, Hbond donor, Rotatable bond, Exact mass, TPSA, Heavy atom count, Log-P
        accelerator.log({"epoch": t, 
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
        
        print(f"Loss: {train_loss[-1]}")
        print(f"Loss per property: {batch_loss_per_prop_mean}")

    accelerator.end_training()
        
    output_list = [train_loss, val_loss_list]
    with open(checkpoint_path+"results_list.pkl", 'wb') as f:
        pickle.dump(output_list, f)
    print("Done!")
                

stop_epoch = 100
if __name__ == "__main__":
    training_loop(tokenizer = smiles_tokenizer, files_index = ifile,
                  properties_tensor = properties_tensor, median = median, Q1 = Q1, Q3 = Q3, 
                  start_epoch = 0, stop_epoch = stop_epoch, d_model = 90, num_samples = num_samples,
                  n_layers = 4, heads = 6, dropout = 0.1, seq_length = 384, num_properties = 7, num_workers = 10,
                  batch_size = 64, lr = 1e-4, betas = (0.9, 0.999), weight_decay = 0.01, warmup_steps=10000,
                  checkpoint_path = "Training_results/",
                  device = 'cuda')
