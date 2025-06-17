import os
import numpy as np 
import torch
import torch.optim as optim

from src.utils.GrOWL_utils import sparsity_info, similarity_info, metrics_tr

###################################
####### MULTI-TASK Training #######
###################################

from src.utils.WCsAL_utils import inner_optimization

from src.utils.lsuv import lsuv_init

def train_multitask_model(train_loader, val_loader, model,
                          params_init, init_model = True):
    
    device, w, a, epsilon = params_init["device"], params_init["w"], params_init["a"], params_init["epsilon"]
    num_tasks, num_outs = params_init["num_tasks"], params_init["num_outs"] #, params_init["num_batchEpoch"]
    max_iter_retrain, max_iter_search, num_epochs, tol_epochs = params_init["max_iter_retrain"], params_init["max_iter_search"], params_init["num_epochs"], params_init["tol_epochs"]
    num_model, main_dir, mod_logdir = params_init["num_model"], params_init["main_dir"], params_init["mod_logdir"]
    is_search, min_sparsRate = params_init["is_search"], params_init["min_sparsRate"]
    
    num_epochs_search, num_epochs_retrain = params_init["num_epochs_search"], params_init["num_epochs_retrain"]
    
    # file names
    if not os.path.exists("%s/%s"%(main_dir, mod_logdir)):
            os.makedirs("%s/%s"%(main_dir, mod_logdir))
    MODEL_FILE = str("%s/%s/model%03d.pth"%(main_dir, mod_logdir, num_model))
    DRAFT_MODEL_FILE = str("%s/%s/draft_model%03d.pth"%(main_dir, mod_logdir, num_model))

    # global variables
    violation_epochs = 0
    
    #--------------------------------------------------------------------------#
    # START ALGORITHM                                                          #
    #--------------------------------------------------------------------------#
    
    #### INITIALIZATION
    mu, rho = params_init["mu"], params_init["rho"]
    #### STARTING POINT
    max_iter = params_init["max_iter"]
    if params_init["w"][0] > 0 and params_init["Sparsity_study"]:
        if is_search:
            max_iter = max_iter_search
            num_epochs = num_epochs_search
            # Create the model
            model = model.to(device)
            # Initialize the networks
            if init_model:
                model = lsuv_init(model, train_loader, needed_std=1.0, std_tol=0.1,
                                max_attempts=10, do_orthonorm=True, device=device) # initialize the model weights
        else:
            max_iter = max_iter_retrain
            num_epochs = num_epochs_retrain
            
    else:
        model = model.to(device)

    # Initialize the Langrange nultiplier lambda
    
    lmbd = torch.ones(num_tasks, requires_grad = False)/num_tasks

    #### LOOP
    ALL_TRAIN_LOSS = []
    ALL_VAL_ACCU = []
    ALL_ORIG_losses = []
    MODEL_VAL_ACCU = []
    best_avg_val_accu = 0.0
    BEST_val_accu = 0.0
    k = 0
    
    lr, lr_sched_coef = params_init["lr"], params_init["lr_sched_coef"]
    base_optimizer = params_init["base_optimizer"]
    LR_scheduler = params_init["LR_scheduler"]
    criterion = params_init["criterion"]
    optimizer = base_optimizer(model.mt_parameters(), lr=lr)
    if LR_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_sched_coef)
    
    act_bst_accu = 0.0
    best_exist = False
    best_accu_search = 0.0
    succeed_bst = False
    orig_max_layerSRate = model.GrOWL_parameters["max_layerSRate"]
    while(k < max_iter):
        print("-------------------------------------")
        print(f"------ Algorithm Iteration {k+1}/{max_iter} ------")
        print("-------------------------------------")
        if params_init["w"][0] > 0 and params_init["Sparsity_study"] and is_search:
            model.GrOWL_parameters["max_layerSRate"] = (k+1)*orig_max_layerSRate/max_iter
        ### FIRST STEP: find best model weights
        TRAIN_LOSS = []
        VAL_ACCU = []
        ORIG_losses = []
        # epfinal = k*(num_epochs*num_batches + num_batches)
        for i in range(num_epochs):
            print("######################")
            print(f"#### EPOCH No {i+1}/{num_epochs} ####")
            print("######################")
            orig_train_losses, train_loss, val_accuracy, contrs_wc_after = inner_optimization(model, params_init, optimizer, w, a, epsilon, criterion, train_loader, val_loader, device, num_tasks,
                                    mu, lmbd,  act_bst_accu, best_exist)
            
            ORIG_losses.append(orig_train_losses.numpy())
            TRAIN_LOSS.append(train_loss)
            VAL_ACCU.append(val_accuracy.numpy())
            MODEL_VAL_ACCU.append(val_accuracy.mean().item())
            
            if params_init["w"][0] > 0 and params_init["Sparsity_study"]:
                if is_search:
                    sparsity_RT, _ = sparsity_info(model, verbose = False)
                    if (sparsity_RT >= min_sparsRate) and (best_accu_search < val_accuracy.mean().item()):
                            succeed_bst = True
                            best_accu_search = val_accuracy.mean().item()
                            # torch.save(model.state_dict(), MODEL_FILE)
                            torch.save(model, MODEL_FILE)

            if(best_avg_val_accu < val_accuracy.mean().item()):
                if not is_search:
                    succeed_bst = True
                    # torch.save(model.state_dict(), MODEL_FILE)
                    torch.save(model, MODEL_FILE)
                else:
                    # torch.save(model.state_dict(), DRAFT_MODEL_FILE)
                    torch.save(model, DRAFT_MODEL_FILE)
                    
                best_avg_val_accu = val_accuracy.mean().item()
                BEST_val_accu = val_accuracy.numpy()
                BEST_contrs_after_optim = contrs_wc_after
                Best_iter = [k, i]
                act_bst_accu = best_avg_val_accu
                best_exist = True
                violation_epochs = 0
                print("Best global performance (Accuracy)!")
                if (num_tasks-1 <= 3):# print out per-task accuracies for problem with small number of main tasks
                    for i in range(num_tasks-1):
                        print("Accuracy Task {}: {:.04f}%".format(i+1, val_accuracy[i].item()))
            else:
                violation_epochs = violation_epochs + 1
                if tol_epochs is not None:
                    if (violation_epochs > tol_epochs):
                        print(f"No improvement in accuracy after {tol_epochs} more epochs. END!!!")  
                        break;
        
        print("Learning rate used: ", optimizer.param_groups[0]['lr'])
        #--------------------------------------------------------------------------#
        # update learning rate scheduler                                           #
        #--------------------------------------------------------------------------#
        if LR_scheduler:
            lr_scheduler.step()

        ALL_TRAIN_LOSS.append(TRAIN_LOSS)
        ALL_VAL_ACCU.append(VAL_ACCU)
        ALL_ORIG_losses.append(ORIG_losses)
        ### SECOND STEP
        lmbd = lmbd.reshape(-1, 1).to(device) + mu*BEST_contrs_after_optim.reshape(-1, 1).to(device)
            
        print("Penalty coefficient (mu) used: ", mu)
        ### THIRD STEP: Update lambda
        mu = rho*mu
        
        ### STOP after a given "maximum number of iterations" 
        k = k + 1

    #### RETURN FINAL (BEST) MODEL
    if succeed_bst:
        # model.load_state_dict(torch.load(MODEL_FILE))
        model = torch.load(MODEL_FILE, weights_only=False)
    else:
        print("Could not find a model with the required sparsity rate!\n The model with the highest accuracy has been returned!")
        # model.load_state_dict(torch.load(DRAFT_MODEL_FILE))
        model = torch.load(DRAFT_MODEL_FILE, weights_only=False)
        
    tr_metrics, _ = metrics_tr(model, verbose = False)
            
    return model, [succeed_bst, tr_metrics], ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter


def full_training(train_loader, val_loader, model,
                        params_init, init_model = True):
    if params_init["Sparsity_study"]:
        print("################################")
        print(f"#### SPARSITY inducing ... ####")
        print("################################")
    else:
        print("################################")
        print(f"#### TRAINING started ! ####")
        print("################################")
    params_init["is_search"] = True
    model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = train_multitask_model(train_loader, val_loader, model,
                        params_init, init_model = init_model)
    
    if params_init["w"][0] > 0 and params_init["Sparsity_study"]:
        _, ZERO_layers = sparsity_info(model)
        print("Computing similarity matrices . . . ")
        similarity_M = similarity_info(model, zero_layers = ZERO_layers)
        print("Done !")

        params_init["zero_layers"] = ZERO_layers
        params_init["similarity_m"] = similarity_M
    
        from time import time
        # Start timer
        import datetime
        print(datetime.datetime.now())
        t_0 = time()
        
        print("###############################")
        print(f"#### RETRAINING started ! ####")
        print("###############################")
        params_init["is_search"] = False    
        model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter =  train_multitask_model(train_loader, val_loader, model,
                            params_init, init_model = False)
    
        T_1 = time()-t_0
        # Print computation time
        print('\nComputation time for RETRAINING: {} minutes'.format(T_1/60))
        print(datetime.datetime.now())
        
    return model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter