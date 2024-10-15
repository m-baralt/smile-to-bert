import torch
import math
from sklearn.model_selection import StratifiedKFold
from Model.binding_functions import SMILESDataset, SMILESDatasetT, SMILESDatasetPE, SMILESDatasetJ, CosineWithRestarts, FC, FCT
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import torch.nn.functional as F
from  Model.Transformer import create_masks


def calc_accuracy(preds, labels):
    pred_idx = torch.max(preds, 1)[1]
    return (pred_idx == labels).to(torch.float).mean().item()

def print_update(epoch=0, iters=0, mode="TRAIN", loss=0, acc=0, c0=0, c1=0):
    print("{mode:>5}- E:{epoch:2d}- I:{iters:4d} loss:{loss:6.3f}, acc:{acc:6.2f}%, c0:{c0:5.1f}%, c1:{c1:5.1f}%".format(epoch=epoch, iters=iters, mode=mode, loss=loss, acc=acc*100, c0=c0*100, c1=c1*100))

def calc_class_accuracies(preds, labels):
    class_accs = []
    for i in range(int(labels.max().item()) + 1):
        class_idxs = torch.nonzero(labels==i)
        class_labels = labels[class_idxs]
        class_preds = preds[class_idxs]
        class_pred_idx = torch.max(class_preds, -1)[1]
        class_acc = (class_pred_idx == class_labels).to(torch.float).mean().item()
        class_accs.append(class_acc)
    return class_accs


def test_epoch(model, embedding_model, epoch, test_dataloader, criterion, device = "cuda"):
    model.eval()
    embedding_model.eval()
    with torch.no_grad():
        all_preds_cls, all_binding, all_preds_int = [], [], []
        for batch, (X, Y) in tqdm.tqdm(enumerate(test_dataloader)):
            Y = Y.to(device)
            X = X.to(device)
            pred, embed = embedding_model(X)
            #embed = torch.flatten(embed[:,0])
            preds_cls, preds_int = model(embed)
            all_preds_cls.append(preds_cls)
            all_binding.append(Y)
            all_preds_int.append(preds_int)
            
        all_preds_cls = torch.cat(all_preds_cls)
        all_binding = torch.cat(all_binding)
        all_preds_int = torch.cat(all_preds_int)
        
        loss = 0
        
        loss_cls = criterion(all_preds_cls, all_binding).item()
        loss += loss_cls
        
        acc, class_accs = 0, [0, 0]
        
        acc = calc_accuracy(all_preds_cls, all_binding)
        class_accs = calc_class_accuracies(all_preds_cls, all_binding)
        
    print_update(mode="TEST", epoch=epoch, iters=batch+1, loss=loss, acc=acc, c0=class_accs[0], c1=class_accs[1])
    
    return loss, acc, class_accs, all_preds_cls, all_preds_int, all_binding


def train(embedding_model, dataset, tokenizer, seq_len = 384, 
          batch_size = 64, lr=0.001, device = "cuda"):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=808)
    comp_accs, comp_labs, comp_preds = [], [], []
    cv_accs, cv_labs, cv_preds = [], [], []
    PRINT_ITERS = int(math.ceil(len(dataset)*9//10 / batch_size))
    for i, (train_idxs, test_idxs) in enumerate(cv.split(dataset['CanonicalSMILES'], dataset['binding'])):
        model = FC(input_features=90*83).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max=len(dataset)/batch_size)
    
        train_dataset = SMILESDataset(data = dataset, tokenizer = tokenizer, seq_len = seq_len, idx = train_idxs)
        test_dataset = SMILESDataset(data = dataset, tokenizer = tokenizer, seq_len = seq_len, idx = test_idxs)
        train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, batch_size=batch_size, num_workers=10)
        test_dataloader = DataLoader(dataset = test_dataset, shuffle=False, batch_size=batch_size, num_workers=10)
        model.train()
        embedding_model.eval()  
        
        test_losses, test_accs = [], []
        for t in range(0, 10):
            print(f"Epoch {t}\n-------------------------------")
            optimizer.zero_grad()
            batch_loss = 0
            running_preds_cls, running_binding = [], []

            for batch, (X, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                X = X.to(device)
                Y = Y.to(device)
                pred, embed = embedding_model(X)
                preds_cls, _ = model(embed)
                loss_cls = criterion(preds_cls, Y)
        
                running_preds_cls.append(preds_cls)
                running_binding.append(Y)
                
                loss_cls.backward()
                
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
        
                batch_loss += loss_cls.item()

                if (batch+1) % PRINT_ITERS == 0:
                    acc, class_accs = 0, [0, 0]
                    running_preds_cls = torch.cat(running_preds_cls)
                    running_binding = torch.cat(running_binding)
                    acc = calc_accuracy(running_preds_cls, running_binding)
                    class_accs = calc_class_accuracies(running_preds_cls, running_binding)
                    print_update(epoch=t, iters=batch+1, loss=batch_loss/PRINT_ITERS, 
                                 acc=acc, c0=class_accs[0], c1=class_accs[1])            
                    batch_loss = 0
                    running_preds_cls, running_binding = [], []
                    
            loss, acc, class_accs, all_preds, all_int, all_labels = test_epoch(model=model, 
                                                                               embedding_model=embedding_model, 
                                                                               epoch = t, 
                                                                               test_dataloader = test_dataloader,
                                                                               criterion = criterion, 
                                                                               device = device)
            test_losses.append(loss)
            test_accs.append(acc)
            if t > 2:
                comp = np.array(test_losses)
                comp = (comp[1:] - comp[:-1]) <= 0
                if not np.any(comp[-2:]):
                    test_accs[-1] = test_accs[-3]
                    break
            
        cv_accs.append(test_accs[-1])
            
        loss, accuracy, class_accs, all_preds, all_int, all_labels = test_epoch(model=model, 
                                                                                embedding_model=embedding_model,
                                                                                epoch = t, 
                                                                                test_dataloader = test_dataloader,
                                                                                criterion = criterion, 
                                                                                device = device)
        all_labels, all_preds = all_labels.cpu().numpy(), all_preds.cpu().numpy()
        all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()[:,1]
        min_test_idxs = int(len(test_idxs)/10)
        cv_labs.append(all_labels)
        cv_preds.append(all_preds)

    return cv_accs, cv_labs, cv_preds

def test_epochPE(model, embedding_model, epoch, test_dataloader, criterion, device = "cuda"):
    model.eval()
    embedding_model.eval()
    with torch.no_grad():
        all_preds_cls, all_binding, all_preds_int = [], [], []
        for batch, (X, Y) in tqdm.tqdm(enumerate(test_dataloader)):
            Y = Y.to(device)
            X = X.to(device)
            pred, embed = embedding_model(X)
            #embed = torch.flatten(embed[:,0])
            preds_cls, preds_int = model(embed)
            all_preds_cls.append(preds_cls)
            all_binding.append(Y)
            all_preds_int.append(preds_int)
            
        all_preds_cls = torch.cat(all_preds_cls)
        all_binding = torch.cat(all_binding)
        all_preds_int = torch.cat(all_preds_int)
        
        loss = 0
        
        loss_cls = criterion(all_preds_cls, all_binding).item()
        loss += loss_cls
        
        acc, class_accs = 0, [0, 0]
        
        acc = calc_accuracy(all_preds_cls, all_binding)
        class_accs = calc_class_accuracies(all_preds_cls, all_binding)
        
    print_update(mode="TEST", epoch=epoch, iters=batch+1, loss=loss, acc=acc, c0=class_accs[0], c1=class_accs[1])
    
    return loss, acc, class_accs, all_preds_cls, all_preds_int, all_binding


def trainPE(embedding_model, dataset, tokenizer, spe, seq_len = 384, 
          batch_size = 64, lr=0.001, device = "cuda"):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=808)
    comp_accs, comp_labs, comp_preds = [], [], []
    cv_accs, cv_labs, cv_preds = [], [], []
    PRINT_ITERS = int(math.ceil(len(dataset)*9//10 / batch_size))
    for i, (train_idxs, test_idxs) in enumerate(cv.split(dataset['CanonicalSMILES'], dataset['binding'])):
        model = FC(input_features=90*30).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max=len(dataset)/batch_size)
    
        train_dataset = SMILESDatasetPE(data = dataset, tokenizer = tokenizer, spe = spe,
                                        seq_len = seq_len, idx = train_idxs)
        test_dataset = SMILESDatasetPE(data = dataset, tokenizer = tokenizer, spe = spe,
                                       seq_len = seq_len, idx = test_idxs)
        train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, 
                                      batch_size=batch_size, num_workers=10)
        test_dataloader = DataLoader(dataset = test_dataset, shuffle=False, 
                                     batch_size=batch_size, num_workers=10)
        model.train()
        embedding_model.eval()  
        
        test_losses, test_accs = [], []
        for t in range(0, 10):
            print(f"Epoch {t}\n-------------------------------")
            optimizer.zero_grad()
            batch_loss = 0
            running_preds_cls, running_binding = [], []

            for batch, (X, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                X = X.to(device)
                Y = Y.to(device)
                pred, embed = embedding_model(X)
                preds_cls, _ = model(embed)
                loss_cls = criterion(preds_cls, Y)
        
                running_preds_cls.append(preds_cls)
                running_binding.append(Y)
                
                loss_cls.backward()
                
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
        
                batch_loss += loss_cls.item()

                if (batch+1) % PRINT_ITERS == 0:
                    acc, class_accs = 0, [0, 0]
                    running_preds_cls = torch.cat(running_preds_cls)
                    running_binding = torch.cat(running_binding)
                    acc = calc_accuracy(running_preds_cls, running_binding)
                    class_accs = calc_class_accuracies(running_preds_cls, running_binding)
                    print_update(epoch=t, iters=batch+1, loss=batch_loss/PRINT_ITERS, 
                                 acc=acc, c0=class_accs[0], c1=class_accs[1])            
                    batch_loss = 0
                    running_preds_cls, running_binding = [], []
                    
            loss, acc, class_accs, all_preds, all_int, all_labels = test_epochPE(model=model, 
                                                                               embedding_model=embedding_model, 
                                                                               epoch = t, 
                                                                               test_dataloader = test_dataloader,
                                                                               criterion = criterion, 
                                                                               device = device)
            test_losses.append(loss)
            test_accs.append(acc)
            if t > 2:
                comp = np.array(test_losses)
                comp = (comp[1:] - comp[:-1]) <= 0
                if not np.any(comp[-2:]):
                    test_accs[-1] = test_accs[-3]
                    break
            
        cv_accs.append(test_accs[-1])
            
        loss, accuracy, class_accs, all_preds, all_int, all_labels = test_epochPE(model=model, 
                                                                                embedding_model=embedding_model,
                                                                                epoch = t, 
                                                                                test_dataloader = test_dataloader,
                                                                                criterion = criterion, 
                                                                                device = device)
        all_labels, all_preds = all_labels.cpu().numpy(), all_preds.cpu().numpy()
        all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()[:,1]
        min_test_idxs = int(len(test_idxs)/10)
        cv_labs.append(all_labels)
        cv_preds.append(all_preds)

    return cv_accs, cv_labs, cv_preds

def trainPE_large(embedding_model, dataset, tokenizer, spe, seq_len = 384, n_tokens = 30,
          batch_size = 64, lr=0.001, device = "cuda"):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=808)
    comp_accs, comp_labs, comp_preds = [], [], []
    cv_accs, cv_labs, cv_preds = [], [], []
    PRINT_ITERS = int(math.ceil(len(dataset)*9//10 / batch_size))
    for i, (train_idxs, test_idxs) in enumerate(cv.split(dataset['CanonicalSMILES'], dataset['binding'])):
        model = FC(input_features=510*n_tokens).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max=len(dataset)/batch_size)
    
        train_dataset = SMILESDatasetPE(data = dataset, tokenizer = tokenizer, spe = spe,
                                        seq_len = seq_len, idx = train_idxs)
        test_dataset = SMILESDatasetPE(data = dataset, tokenizer = tokenizer, spe = spe,
                                       seq_len = seq_len, idx = test_idxs)
        train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, 
                                      batch_size=batch_size, num_workers=10)
        test_dataloader = DataLoader(dataset = test_dataset, shuffle=False, 
                                     batch_size=batch_size, num_workers=10)
        model.train()
        embedding_model.eval()  
        
        test_losses, test_accs = [], []
        for t in range(0, 10):
            print(f"Epoch {t}\n-------------------------------")
            optimizer.zero_grad()
            batch_loss = 0
            running_preds_cls, running_binding = [], []

            for batch, (X, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                X = X.to(device)
                Y = Y.to(device)
                pred, embed = embedding_model(X)
                preds_cls, _ = model(embed)
                loss_cls = criterion(preds_cls, Y)
        
                running_preds_cls.append(preds_cls)
                running_binding.append(Y)
                
                loss_cls.backward()
                
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
        
                batch_loss += loss_cls.item()

                if (batch+1) % PRINT_ITERS == 0:
                    acc, class_accs = 0, [0, 0]
                    running_preds_cls = torch.cat(running_preds_cls)
                    running_binding = torch.cat(running_binding)
                    acc = calc_accuracy(running_preds_cls, running_binding)
                    class_accs = calc_class_accuracies(running_preds_cls, running_binding)
                    print_update(epoch=t, iters=batch+1, loss=batch_loss/PRINT_ITERS, 
                                 acc=acc, c0=class_accs[0], c1=class_accs[1])            
                    batch_loss = 0
                    running_preds_cls, running_binding = [], []
                    
            loss, acc, class_accs, all_preds, all_int, all_labels = test_epochPE(model=model, 
                                                                               embedding_model=embedding_model, 
                                                                               epoch = t, 
                                                                               test_dataloader = test_dataloader,
                                                                               criterion = criterion, 
                                                                               device = device)
            test_losses.append(loss)
            test_accs.append(acc)
            if t > 2:
                comp = np.array(test_losses)
                comp = (comp[1:] - comp[:-1]) <= 0
                if not np.any(comp[-2:]):
                    test_accs[-1] = test_accs[-3]
                    break
            
        cv_accs.append(test_accs[-1])
            
        loss, accuracy, class_accs, all_preds, all_int, all_labels = test_epochPE(model=model, 
                                                                                embedding_model=embedding_model,
                                                                                epoch = t, 
                                                                                test_dataloader = test_dataloader,
                                                                                criterion = criterion, 
                                                                                device = device)
        all_labels, all_preds = all_labels.cpu().numpy(), all_preds.cpu().numpy()
        all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()[:,1]
        min_test_idxs = int(len(test_idxs)/10)
        cv_labs.append(all_labels)
        cv_preds.append(all_preds)

    return cv_accs, cv_labs, cv_preds


def test_epochT(model, embedding_model, epoch, test_dataloader, criterion, device = "cuda"):
    model.eval()
    embedding_model.eval()
    with torch.no_grad():
        all_preds_cls, all_binding, all_preds_int = [], [], []
        for batch, (X, Y) in tqdm.tqdm(enumerate(test_dataloader)):
            Y = Y.to(device)
            X = X.to(device)
            mask = create_masks(X)
            embed = embedding_model.module.encoder(X, mask)
            #embed = torch.flatten(embed[:,0])
            preds_cls, preds_int = model(embed)
            all_preds_cls.append(preds_cls)
            all_binding.append(Y)
            all_preds_int.append(preds_int)
            
        all_preds_cls = torch.cat(all_preds_cls)
        all_binding = torch.cat(all_binding)
        all_preds_int = torch.cat(all_preds_int)
        
        loss = 0
        
        loss_cls = criterion(all_preds_cls, all_binding).item()
        loss += loss_cls
        
        acc, class_accs = 0, [0, 0]
        
        acc = calc_accuracy(all_preds_cls, all_binding)
        class_accs = calc_class_accuracies(all_preds_cls, all_binding)
        
    print_update(mode="TEST", epoch=epoch, iters=batch+1, loss=loss, acc=acc, c0=class_accs[0], c1=class_accs[1])
    
    return loss, acc, class_accs, all_preds_cls, all_preds_int, all_binding


def trainT(embedding_model, dataset, batch_size = 64, lr=0.001, device = "cuda"):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=808)
    comp_accs, comp_labs, comp_preds = [], [], []
    cv_accs, cv_labs, cv_preds = [], [], []
    PRINT_ITERS = int(math.ceil(len(dataset)*9//10 / batch_size))
    for i, (train_idxs, test_idxs) in enumerate(cv.split(dataset['CanonicalSMILES'], dataset['binding'])):
        model = FCT(input_features=512*256).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max=len(dataset)/batch_size)
    
        train_dataset = SMILESDatasetT(data = dataset, idx = train_idxs)
        test_dataset = SMILESDatasetT(data = dataset, idx = test_idxs)
        train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, batch_size=batch_size, num_workers=10)
        test_dataloader = DataLoader(dataset = test_dataset, shuffle=False, batch_size=batch_size, num_workers=10)
        model.train()
        embedding_model.eval()  
        
        test_losses, test_accs = [], []
        for t in range(0, 10):
            print(f"Epoch {t}\n-------------------------------")
            optimizer.zero_grad()
            batch_loss = 0
            running_preds_cls, running_binding = [], []

            for batch, (X, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                X = X.to(device)
                Y = Y.to(device)
                mask = create_masks(X)
                embed = embedding_model.module.encoder(X, mask)
                preds_cls, _ = model(embed)
                loss_cls = criterion(preds_cls, Y)
        
                running_preds_cls.append(preds_cls)
                running_binding.append(Y)
                
                loss_cls.backward()
                
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
        
                batch_loss += loss_cls.item()

                if (batch+1) % PRINT_ITERS == 0:
                    acc, class_accs = 0, [0, 0]
                    running_preds_cls = torch.cat(running_preds_cls)
                    running_binding = torch.cat(running_binding)
                    acc = calc_accuracy(running_preds_cls, running_binding)
                    class_accs = calc_class_accuracies(running_preds_cls, running_binding)
                    print_update(epoch=t, iters=batch+1, loss=batch_loss/PRINT_ITERS, 
                                 acc=acc, c0=class_accs[0], c1=class_accs[1])            
                    batch_loss = 0
                    running_preds_cls, running_binding = [], []
                    
            loss, acc, class_accs, all_preds, all_int, all_labels = test_epochT(model=model, 
                                                                                embedding_model=embedding_model,
                                                                                epoch = t, 
                                                                                test_dataloader = test_dataloader,
                                                                                criterion = criterion, 
                                                                                device = device)
            test_losses.append(loss)
            test_accs.append(acc)
            if t > 2:
                comp = np.array(test_losses)
                comp = (comp[1:] - comp[:-1]) <= 0
                if not np.any(comp[-2:]):
                    test_accs[-1] = test_accs[-3]
                    break
            
        cv_accs.append(test_accs[-1])
            
        loss, accuracy, class_accs, all_preds, all_int, all_labels = test_epochT(model=model, 
                                                                                 embedding_model=embedding_model, 
                                                                                 epoch = t, 
                                                                                 test_dataloader = test_dataloader,
                                                                                 criterion = criterion, 
                                                                                 device = device)
        all_labels, all_preds = all_labels.cpu().numpy(), all_preds.cpu().numpy()
        all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()[:,1]
        min_test_idxs = int(len(test_idxs)/10)
        cv_labs.append(all_labels)
        cv_preds.append(all_preds)

    return cv_accs, cv_labs, cv_preds


def test_epochF(model, embedding_model, epoch, test_dataloader, criterion, device = "cuda"):
    model.eval()
    embedding_model.eval()
    with torch.no_grad():
        all_preds_cls, all_binding, all_preds_int = [], [], []
        for batch, (X, Y) in tqdm.tqdm(enumerate(test_dataloader)):
            Y = Y.to(device)
            X = X.to(device)
            mask = create_masks(X)
            embed = embedding_model.encoder(X, mask)
            #embed = torch.flatten(embed[:,0])
            preds_cls, preds_int = model(embed)
            all_preds_cls.append(preds_cls)
            all_binding.append(Y)
            all_preds_int.append(preds_int)
            
        all_preds_cls = torch.cat(all_preds_cls)
        all_binding = torch.cat(all_binding)
        all_preds_int = torch.cat(all_preds_int)
        
        loss = 0
        
        loss_cls = criterion(all_preds_cls, all_binding).item()
        loss += loss_cls
        
        acc, class_accs = 0, [0, 0]
        
        acc = calc_accuracy(all_preds_cls, all_binding)
        class_accs = calc_class_accuracies(all_preds_cls, all_binding)
        
    print_update(mode="TEST", epoch=epoch, iters=batch+1, loss=loss, acc=acc, c0=class_accs[0], c1=class_accs[1])
    
    return loss, acc, class_accs, all_preds_cls, all_preds_int, all_binding


def trainF(embedding_model, dataset, batch_size = 64, lr=0.001, device = "cuda"):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=808)
    comp_accs, comp_labs, comp_preds = [], [], []
    cv_accs, cv_labs, cv_preds = [], [], []
    PRINT_ITERS = int(math.ceil(len(dataset)*9//10 / batch_size))
    for i, (train_idxs, test_idxs) in enumerate(cv.split(dataset['CanonicalSMILES'], dataset['binding'])):
        model = FCT(input_features=512*256).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max=len(dataset)/batch_size)
    
        train_dataset = SMILESDatasetT(data = dataset, idx = train_idxs)
        test_dataset = SMILESDatasetT(data = dataset, idx = test_idxs)
        train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, batch_size=batch_size, num_workers=10)
        test_dataloader = DataLoader(dataset = test_dataset, shuffle=False, batch_size=batch_size, num_workers=10)
        model.train()
        embedding_model.eval()  
        
        test_losses, test_accs = [], []
        for t in range(0, 10):
            print(f"Epoch {t}\n-------------------------------")
            optimizer.zero_grad()
            batch_loss = 0
            running_preds_cls, running_binding = [], []

            for batch, (X, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                X = X.to(device)
                Y = Y.to(device)
                mask = create_masks(X)
                embed = embedding_model.encoder(X, mask)
                preds_cls, _ = model(embed)
                loss_cls = criterion(preds_cls, Y)
        
                running_preds_cls.append(preds_cls)
                running_binding.append(Y)
                
                loss_cls.backward()
                
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
        
                batch_loss += loss_cls.item()

                if (batch+1) % PRINT_ITERS == 0:
                    acc, class_accs = 0, [0, 0]
                    running_preds_cls = torch.cat(running_preds_cls)
                    running_binding = torch.cat(running_binding)
                    acc = calc_accuracy(running_preds_cls, running_binding)
                    class_accs = calc_class_accuracies(running_preds_cls, running_binding)
                    print_update(epoch=t, iters=batch+1, loss=batch_loss/PRINT_ITERS, 
                                 acc=acc, c0=class_accs[0], c1=class_accs[1])            
                    batch_loss = 0
                    running_preds_cls, running_binding = [], []
                    
            loss, acc, class_accs, all_preds, all_int, all_labels = test_epochF(model=model, 
                                                                                embedding_model=embedding_model,
                                                                                epoch = t, 
                                                                                test_dataloader = test_dataloader,
                                                                                criterion = criterion, 
                                                                                device = device)
            test_losses.append(loss)
            test_accs.append(acc)
            if t > 2:
                comp = np.array(test_losses)
                comp = (comp[1:] - comp[:-1]) <= 0
                if not np.any(comp[-2:]):
                    test_accs[-1] = test_accs[-3]
                    break
            
        cv_accs.append(test_accs[-1])
            
        loss, accuracy, class_accs, all_preds, all_int, all_labels = test_epochF(model=model, 
                                                                                 embedding_model=embedding_model, 
                                                                                 epoch = t, 
                                                                                 test_dataloader = test_dataloader,
                                                                                 criterion = criterion, 
                                                                                 device = device)
        all_labels, all_preds = all_labels.cpu().numpy(), all_preds.cpu().numpy()
        all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()[:,1]
        min_test_idxs = int(len(test_idxs)/10)
        cv_labs.append(all_labels)
        cv_preds.append(all_preds)

    return cv_accs, cv_labs, cv_preds


##############################

def test_epochJ(model, spe_model, transformer, epoch, test_dataloader, criterion, device = "cuda"):
    model.eval()
    spe_model.eval()
    transformer.eval()  
    with torch.no_grad():
        all_preds_cls, all_binding, all_preds_int = [], [], []
        for batch, (Xt, Xspe, Y) in tqdm.tqdm(enumerate(test_dataloader)):
            Xt = Xt.to(device)
            Xspe = Xspe.to(device)
            Y = Y.to(device)
            
            mask = create_masks(Xt)

            pred, embed_spe = spe_model(Xspe)
            embed_trans = transformer.module.encoder(Xt, mask)
            embed_trans = embed_trans.view(-1, 512*256) 

            embed = torch.cat((embed_trans, embed_spe), dim=1)
            #embed = torch.flatten(embed[:,0])
            preds_cls, preds_int = model(embed)
            all_preds_cls.append(preds_cls)
            all_binding.append(Y)
            all_preds_int.append(preds_int)
            
        all_preds_cls = torch.cat(all_preds_cls)
        all_binding = torch.cat(all_binding)
        all_preds_int = torch.cat(all_preds_int)
        
        loss = 0
        
        loss_cls = criterion(all_preds_cls, all_binding).item()
        loss += loss_cls
        
        acc, class_accs = 0, [0, 0]
        
        acc = calc_accuracy(all_preds_cls, all_binding)
        class_accs = calc_class_accuracies(all_preds_cls, all_binding)
        
    print_update(mode="TEST", epoch=epoch, iters=batch+1, loss=loss, acc=acc, c0=class_accs[0], c1=class_accs[1])
    
    return loss, acc, class_accs, all_preds_cls, all_preds_int, all_binding


def trainJ(spe_model, transformer, dataset, tokenizer, spe, seq_len = 384, 
          batch_size = 64, lr=0.001, device = "cuda"):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=808)
    comp_accs, comp_labs, comp_preds = [], [], []
    cv_accs, cv_labs, cv_preds = [], [], []
    PRINT_ITERS = int(math.ceil(len(dataset)*9//10 / batch_size))

    
    for i, (train_idxs, test_idxs) in enumerate(cv.split(dataset['CanonicalSMILES'], dataset['binding'])):
        model = FC(input_features=(512*256)+(510*113)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max=len(dataset)/batch_size)

        # x = x.view(-1, self.input_size) 
        train_dataset = SMILESDatasetJ(data = dataset, idx = train_idxs, tokenizer = tokenizer, 
                                       spe = spe, seq_len = seq_len)
        test_dataset = SMILESDatasetJ(data = dataset, idx = test_idxs, tokenizer = tokenizer, 
                                       spe = spe, seq_len = seq_len)
        train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, 
                                      batch_size=batch_size, num_workers=10)
        test_dataloader = DataLoader(dataset = test_dataset, shuffle=False, 
                                     batch_size=batch_size, num_workers=10)
        model.train()
        spe_model.eval()  
        transformer.eval()  
        
        test_losses, test_accs = [], []
        for t in range(0, 10):
            print(f"Epoch {t}\n-------------------------------")
            optimizer.zero_grad()
            batch_loss = 0
            running_preds_cls, running_binding = [], []

            for batch, (Xt, Xspe, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                Xt = Xt.to(device)
                Xspe = Xspe.to(device)
                Y = Y.to(device)
                mask = create_masks(Xt)

                pred, embed_spe = spe_model(Xspe)
                embed_trans = transformer.module.encoder(Xt, mask)
                embed_trans = embed_trans.view(-1, 512*256) 

                embed = torch.cat((embed_trans, embed_spe), dim=1)
                
                preds_cls, _ = model(embed)
                loss_cls = criterion(preds_cls, Y)
        
                running_preds_cls.append(preds_cls)
                running_binding.append(Y)
                
                loss_cls.backward()
                
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
        
                batch_loss += loss_cls.item()

                if (batch+1) % PRINT_ITERS == 0:
                    acc, class_accs = 0, [0, 0]
                    running_preds_cls = torch.cat(running_preds_cls)
                    running_binding = torch.cat(running_binding)
                    acc = calc_accuracy(running_preds_cls, running_binding)
                    class_accs = calc_class_accuracies(running_preds_cls, running_binding)
                    print_update(epoch=t, iters=batch+1, loss=batch_loss/PRINT_ITERS, 
                                 acc=acc, c0=class_accs[0], c1=class_accs[1])            
                    batch_loss = 0
                    running_preds_cls, running_binding = [], []
                    
            loss, acc, class_accs, all_preds, all_int, all_labels = test_epochJ(model=model, 
                                                                               spe_model = spe_model, transformer = transformer,
                                                                               epoch = t, 
                                                                               test_dataloader = test_dataloader,
                                                                               criterion = criterion, 
                                                                               device = device)
            test_losses.append(loss)
            test_accs.append(acc)
            if t > 2:
                comp = np.array(test_losses)
                comp = (comp[1:] - comp[:-1]) <= 0
                if not np.any(comp[-2:]):
                    test_accs[-1] = test_accs[-3]
                    break
            
        cv_accs.append(test_accs[-1])
            
        loss, accuracy, class_accs, all_preds, all_int, all_labels = test_epochJ(model=model, 
                                                                                spe_model = spe_model, transformer = transformer,
                                                                                epoch = t, 
                                                                                test_dataloader = test_dataloader,
                                                                                criterion = criterion, 
                                                                                device = device)
        all_labels, all_preds = all_labels.cpu().numpy(), all_preds.cpu().numpy()
        all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()[:,1]
        min_test_idxs = int(len(test_idxs)/10)
        cv_labs.append(all_labels)
        cv_preds.append(all_preds)


    return cv_accs, cv_labs, cv_preds



