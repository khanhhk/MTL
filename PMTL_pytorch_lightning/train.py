import torch
import pytorch_lightning as pl
import numpy as np
from torch.autograd import Variable
from model.lenet import RegressionTrain, RegressionModel
from utils import circle_points, get_d_paretomtl

class Model_training(pl.LightningModule):
    def __init__(self, n_tasks, init_weight, npref, pref_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tasks = n_tasks
        self.init_weight = init_weight
        self.npref = npref
        self.pref_idx = pref_idx
        
        self.model = RegressionTrain(RegressionModel(self.n_tasks), self.init_weight)
        self.ref_vec = torch.tensor(circle_points([1], [self.npref])[0]).cuda().float()

        self.automatic_optimization = False
        
    def forward(self, x, ts):
        return self.model(x, ts)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    
    def training_step(self, batch, batch_idx):
        X, ts = batch
        optimizer = self.optimizers()
        grads = {}
        losses_vec = []
        
        for i in range(self.n_tasks):
            optimizer.zero_grad()
            task_loss = self.model(X, ts) 
            losses_vec.append(task_loss[i].data)
            
            task_loss[i].backward()
        
            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment              
            grads[i] = []
            for param in self.parameters():
                if param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        grads = torch.stack(grads_list)
        
        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        weight_vec = get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)
        
        normalize_coeff = self.n_tasks / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff
        
        # optimization step
        optimizer.zero_grad()
        for i in range(len(task_loss)):
            task_loss = self.model(X, ts)
            if i == 0:
                loss_total = weight_vec[i] * task_loss[i]
            else:
                loss_total = loss_total + weight_vec[i] * task_loss[i]
                
        self.manual_backward(loss_total)
        optimizer.step()

        self.log_dict({"train_loss": loss_total}, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        X, ts = batch

        # valid_train_loss = model(X, ts)
        output1 = self.model.model(X).max(2, keepdim=True)[1][:,0]
        output2 = self.model.model(X).max(2, keepdim=True)[1][:,1]

        val_acc_batch = np.stack([1.0 * output1.eq(ts[:,0].view_as(output1)).sum().item() / len(batch),
                                  1.0 * output2.eq(ts[:,1].view_as(output2)).sum().item() / len(batch)])
        self.log_dict({"val_acc1": val_acc_batch[0], "val_acc2": val_acc_batch[1]}, prog_bar=True)