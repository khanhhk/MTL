import torch
import os
import pickle
from optimizer import build_MGDA_optimizer
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from load_data import load_MultiMnist_mgda
from utils import train_multi, test_multi

def train_test_MGDA(model, n_tasks, data_name, mod_params_mgda, device):
    training_epochs = mod_params_mgda["training_epochs"]
    batch_size = mod_params_mgda["batch_size"]
    momentum = mod_params_mgda["momentum"]
    lr = mod_params_mgda["lr"]

    if data_name == "MultiMnist":
        X_train, X_test, y_train, y_test, _, _ = load_MultiMnist_mgda()
    else: raise ValueError(f"Unknown dataset {data_name} !")
    
    train_losses = []
    test_accuracies = []

    from time import time
    import datetime
    print("\nStart training at:", datetime.datetime.now())
    t0 = time()
    
    model_multi = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    MTLOptimizerClass = build_MGDA_optimizer(torch.optim.SGD)
    mtl_optim = MTLOptimizerClass(model_multi.parameters(), lr=lr, momentum=momentum)

    X_train = X_train.clone().detach().to(device).float()
    y_train = y_train.clone().detach().to(device).long()

    for epoch in tqdm(range(training_epochs)):
        #print("Training...")
        model_multi.train()
        train_loss = train_multi(X_train, y_train, model_multi, n_tasks, mtl_optim, loss_fn, batch_size[0], device=device)
        train_losses.extend(train_loss)

        # Halve learning rate every 30 epochs
        if epoch > 0 and epoch % 30 == 0:
            for optim_param in mtl_optim.param_groups:
                optim_param['lr'] = optim_param['lr'] / 2

    print("Testing...")
    model_multi.eval()

    X_test = X_test.clone().detach().to(device).float()
    y_test = y_test.clone().detach().to(device).long()

    test_acc_task = test_multi(
        X_test,
        y_test,
        model_multi,
        n_tasks,
        batch_size[1],
        device=device
    )

    test_accu_t1 = 100*sum(test_acc_task[0])/len(test_acc_task[0])
    test_accu_t2 = 100*sum(test_acc_task[1])/len(test_acc_task[1])

    print(f"Test Accuracy Task 1: {test_accu_t1:.2f}%")
    print(f"Test Accuracy Task 2: {test_accu_t2:.2f}%")

    test_accuracies.append([test_accu_t1, test_accu_t2])

    T_norm_1 = time()-t0
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())
    
    os.makedirs('logs', exist_ok=True)
    torch.save(model_multi.model.state_dict(), 'logs/model_mtl.pickle')
    with open(f'logs/MultiMnist_results.pkl', 'wb') as f:
        pickle.dump((train_losses, test_accuracies), f)
    return train_losses, test_accuracies 
