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
from Model.BERT import BERT, SMILESLM, SMILESDatasetPE
import pandas as pd
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--wandb_entity", type=str, help="Wandb username.")
args = parser.parse_args()

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


def prepare_data(data, tokenizer, spe, batch_size, seq_len, train_perc, num_workers, seed=10):
    #np.random.seed(seed)
    num_total = len(data)
    train_size = int(num_total * train_perc)
    
    indices = np.random.permutation(num_total)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Use train indices to extract only training data
    train_properties = data.iloc[train_indices,1:]
    train_properties = torch.tensor(train_properties.values, dtype=torch.float32)
    
    # Compute statistics only on training data
    median = torch.median(train_properties, dim=0)[0]
    Q1 = torch.quantile(train_properties, 0.25, dim=0)
    Q3 = torch.quantile(train_properties, 0.75, dim=0)

    # Create dataset objects using computed statistics
    train_dataset = SMILESDatasetPE(data = data.iloc[train_indices], tokenizer = tokenizer, 
                                  spe = spe, seq_len = seq_len, 
                                  median = median, Q1 = Q1, Q3 = Q3, change_smi = False)

    test_dataset = SMILESDatasetPE(data = data.iloc[test_indices], tokenizer = tokenizer, 
                                  spe = spe, seq_len = seq_len, 
                                  median = median, Q1 = Q1, Q3 = Q3, change_smi = False)


    # Initialize dataloader
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, 
                                  num_workers=num_workers)

    return train_dataloader, test_dataset, len(train_dataset), median, Q1, Q3




def validation(model, testset, device, median, Q1, Q3, batch_size, num_workers, num_properties):

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
            pred, embed = model(X)
            loss = loss_fn(pred, Y)
            loss_list = loss_per_prop(pred = pred, Y = Y, median = median, 
                                      Q1 = Q1, Q3 = Q3, loss_fn = loss_fn, 
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
    spe,
    data,
    d_model,
    n_layers,
    heads,
    batch_size,
    chosen_descriptors,
    dropout = 0.1,
    seq_length = 100,
    num_properties = 200,
    betas = (0.9, 0.999),
    weight_decay = 0.01,
    warmup_steps=10000,
    start_epoch = 0,
    stop_epoch = 100,
    checkpoint_path = None,
    num_workers = 15,
    device = 'cuda'
):
    
    accelerator = Accelerator(gradient_accumulation_steps=1, log_with="wandb", 
                             kwargs_handlers = dist.init_process_group(backend='nccl', init_method='env://', 
                                                                       timeout=datetime.timedelta(seconds=7200)))
    accelerator.init_trackers(
        project_name="BERT-smiles-training", 
        config={"epochs": stop_epoch},
        init_kwargs={"wandb": {"entity": args.wandb_entity}}
    )

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Model configuration
    vocab_size = len(tokenizer.vocab)
    bert_model = BERT(vocab_size = vocab_size, d_model=d_model, n_layers=n_layers, heads=heads, 
                      dropout=dropout, seq_len = seq_length, device = device)
    smiles_model = SMILESLM(bert_model = bert_model, output = num_properties)
    smiles_model.to(device)
    
    # data preparation
    train_loader, testset, train_size, median, Q1, Q3 = prepare_data(data = data, tokenizer = tokenizer, spe = spe,
                                                                     batch_size = batch_size, seq_len = seq_length, 
                                                                     train_perc = 0.95, 
                                                                     num_workers = num_workers, seed = 2)

    # optimization
    loss_fn = nn.L1Loss().to(device)

    optimizer = torch.optim.AdamW([{'params': smiles_model.bert.parameters(), 'lr': 2e-4, 
                                    'weight_decay': weight_decay},
                                   {'params': smiles_model.propsPrediction.parameters(), 
                                    'lr': 5e-5, 'weight_decay': weight_decay}])

        
    optim_schedule = ScheduledOptim(optimizer = optimizer, d_model = d_model, 
                                    n_warmup_steps=warmup_steps, 
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
    print_every_n_batches = 100
    gradients_per_batch = []

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
                pred, embed = smiles_model(X)
                
                # loss computation
                loss = loss_fn(pred, Y)
                
                # loss per property computation
                loss_list = loss_per_prop(pred = pred, Y = Y, median = median, 
                                          Q1 = Q1, Q3 = Q3, loss_fn = loss_fn, 
                                          num_properties = num_properties)

                #import pdb; pdb.set_trace()
                
                # backward pass
                optim_schedule.zero_grad()
                accelerator.backward(loss)
                

                # optimizer step and scheduler update
                optim_schedule.step_and_update_lr()
                
                # batch losses list appending
                batch_loss = batch_loss+loss.item()
                
                #batch_loss.append(loss.item())
                
                batch_loss_per_prop_temp = []
                for i in range(len(loss_list)):
                    batch_loss_per_prop_temp.append(loss_list[i]+batch_loss_per_prop[i])
                    
                batch_loss_per_prop = batch_loss_per_prop_temp
    
                if ((batch==0) | (batch%100==99)):
                    
                    mean_loss = batch_loss/(batch+1)
                    batch_loss_per_prop_mean = [val/(batch+1) for val in batch_loss_per_prop]


                    # Create dictionary dynamically from chosen_descriptors
                    descriptor_log = {desc: value for desc, value in zip(chosen_descriptors, batch_loss_per_prop_mean)}
                    
                    # Add other fixed values (epoch, batch, and Loss)
                    descriptor_log.update({
                        "epoch": t, 
                        "batch": batch,
                        "Loss": mean_loss
                    })

                    grad_norms = {name: p.grad.norm().item() for name, p in smiles_model.named_parameters() if p.grad is not None}
                    gradients_per_batch.append(grad_norms)
                
                    # Prepare gradient log with the layer names and their gradient norms
                    gradient_log = {
                        f"gradient/{name}": grad_norm for name, grad_norm in grad_norms.items()
                    }
                    descriptor_log.update(gradient_log)
                    # Log everything in wandb
                    accelerator.log(descriptor_log)

                    gc.collect()

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

        # Create dictionary dynamically from chosen_descriptors
        descriptor_log = {f"Epoch {desc}": value for desc, value in zip(chosen_descriptors, batch_loss_per_prop_mean)}
        descriptor_log_val = {f"Validation {desc}": value for desc, value in zip(chosen_descriptors, val_loss_per_prop)}
        descriptor_log.update(descriptor_log_val)
        
        # Add other fixed values (epoch, batch, and Loss)
        descriptor_log.update({
            "epoch": t, 
            "batch": batch,
            "Epoch Loss": train_loss[-1],
            "Epoch validation loss": val_loss,
        })
        
        # Log everything in wandb
        accelerator.log(descriptor_log)
        
        #output_batch_list.append([batch_loss, batch_loss_per_prop])
        print(f"Loss: {train_loss[-1]}")
        print(f"Loss per property: {batch_loss_per_prop_mean}")

    accelerator.end_training()
        
    output_list = [train_loss, val_loss_list]
    with open(checkpoint_path+"results_list.pkl", 'wb') as f:
        pickle.dump(output_list, f)
    print("Done!")
                

stop_epoch = 100
if __name__ == "__main__":
    training_loop(tokenizer = smiles_tokenizer, spe = spe, data = data,
                  start_epoch = 0, stop_epoch = stop_epoch, d_model = 512, 
                  n_layers = 4, heads = 8, dropout = 0.1, seq_length = 100, 
                  num_properties = num_props, num_workers = 10,
                  batch_size = 64, betas = (0.9, 0.999), 
                  weight_decay = 0.01, warmup_steps=2000,
                  chosen_descriptors = chosen_descriptors,
                  checkpoint_path = "results/",
                  device = 'cuda')
