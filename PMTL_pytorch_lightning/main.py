import torch
from torch.autograd import Variable
import numpy as np
import pytorch_lightning as pl
from utils import circle_points, get_d_paretomtl_init
from data.load_data import load_data
from train import Model_training

def train_init(data, model, optimizer, scheduler, npref, pref_idx):

    # generate #npref preference vectors      
    n_tasks = 2
    ref_vec = torch.tensor(circle_points([1], [npref])[0]).cuda().float()
    
    # load dataset  
    train_loader = data.train_dataloader()

    # define the base model for ParetoMTL  
    if torch.cuda.is_available():
        model.cuda()
    
    # print the current preference vector
    print('Preference Vector ({}/{}):'.format(pref_idx + 1, npref))
    print(ref_vec[pref_idx].cpu().numpy())

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found 
    # usually can be found with a few steps
    for t in range(2):
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []
                  
            # obtain and store the gradient value
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
                
                grads[i] = []
                
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
            
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
            
            # early stop once a feasible solution is obtained
            if flag == True:
                print("fealsible solution is obtained.")
                break
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            loss_total.backward()
            optimizer.step()
                
        else:
        # continue if no feasible solution is found
            continue
        # break the loop once a feasible solutions is found
        break

if __name__ == '__main__':
    data = load_data(file_id = '1b4ZjhHC8zSeAjlsaCOu1j6ZMC7G3V9dU')
    model = Model_training(n_tasks=2, init_weight = np.array([0.5 , 0.5 ]), npref=5, pref_idx=1)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)
    train_init(
            data,
            model,
            optimizer = optimizer,
            scheduler = scheduler,
            npref=5, pref_idx=1)
    trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=100)
    
    trainer.fit(model, data)