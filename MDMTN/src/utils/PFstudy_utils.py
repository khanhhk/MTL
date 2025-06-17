import os
import random
import numpy as np 
import torch

import matplotlib.pyplot as plt

from src.utils.GrOWL_utils import get_sparse_model_info
from src.utils.WCsAL_Train import train_multitask_model
from src.utils.WCsAL_Test import test_multitask_model

########################################################################
#####  Helper functions for the additional (2D) Pareto front study #####
########################################################################

def eps_dominance(Obj_space, epsilon, start = 0):
    N = len(Obj_space)
    Pareto_set_idx =  list(range(N))
    Dominated = []
    
    for i in range(N):
        candt = Obj_space[i] - epsilon
        for j in range(N):
            if np.all(candt[start:] >= Obj_space[j][start:]) and np.any(candt[start:] > Obj_space[j][start:]):
                Dominated.append(i)
                break; 
                
    PS_idx = list(set(Pareto_set_idx) - set(Dominated))
    return PS_idx

def Train_Test_PFstudy(ws, train_loader, val_loader, test_loader, params,
                       SPARSE_MODEL_FILE, data_name, archi_name, inst_model):
    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t_start = time()

    params["num_batchEpoch"] = len(train_loader)

    print("\nThis is a Pareto Front Study given a Sparse Model Architecture.")
    print("No Parameter Sharing or additional Sparsification will be induced !\n")

    Best_w = ws[0]
    Best_model_test_accu = 0.0

    dec_test_accu = []
    dec_train_loss = []
    dec_val_accu = []
    dec_val_orig_losses = []
    dec_model_val_accu = []
    dec_best_val_accu = []
    dec_BEST_iter = []
    dec_perc_wrong_pred = []
    dec_alphas = []
    dec_tr_metrics = []

    for w in ws:
        print("*****************************************")
        print(f"****** For k = {w} ******")
        print("*****************************************")
        params["w"] = w
        mod_logdir = "Sparse_"+"_model_k1_"+str(w[0])+"_k2_"+str(w[1])+"_k3_"+str(w[2])
        params["mod_logdir"] = mod_logdir
        
        # # Create an instance of the model
        # model = inst_model
        # model.load_state_dict(torch.load(SPARSE_MODEL_FILE))
        model = torch.load(SPARSE_MODEL_FILE, weights_only=False)
        params["info_sparse_model"] = get_sparse_model_info(model)
        
        # Train model
        model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, BEST_iter = train_multitask_model(train_loader,
                                                                                                                    val_loader,
                                                                                                                    model, params,
                                                                                                                    init_model = False)
        dec_train_loss.append(ALL_TRAIN_LOSS)
        dec_val_accu.append(ALL_VAL_ACCU)
        dec_val_orig_losses.append(ALL_ORIG_losses)
        dec_model_val_accu.append(MODEL_VAL_ACCU)
        dec_best_val_accu.append(BEST_val_accu)
        dec_BEST_iter.append(BEST_iter)
        dec_alphas.append(model.get_alphas())
        dec_tr_metrics.append(TR_metrics)
        
        #print("Alpha used: ", model.get_alphas())
        
        test_accuracy, perc_wrong_pred = test_multitask_model(test_loader, model, params, TR_metrics)
        dec_test_accu.append(test_accuracy)
        dec_perc_wrong_pred.append(perc_wrong_pred)
        
        if test_accuracy.mean().item() >= Best_model_test_accu:
            Best_model_test_accu = test_accuracy.mean().item()
            Best_w = w
            print("Actual Best model found !!! k = ", Best_w)
            
        print(f"Best model so far: k = {Best_w} ({Best_model_test_accu}%)")
        
    T_norm_1 = time()-t_start
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print(f"The best Model is obtained using k = {Best_w}. Its test accuracy is: {Best_model_test_accu}")

    ind_best_w = torch.where(torch.all(torch.tensor(ws) == torch.tensor(Best_w), dim=1))
    ind_best_w = ind_best_w[0].item()

    import pickle

    # ## Save the lists to a file
    with open(f'{params["main_dir"]}/PFstudy_results_k0is_{str(Best_w[0])}.pkl', 'wb') as f:
        pickle.dump((ws, dec_train_loss, dec_val_accu, dec_val_orig_losses,
                     dec_model_val_accu, dec_best_val_accu, dec_test_accu,
                     dec_BEST_iter, dec_perc_wrong_pred, dec_alphas, dec_tr_metrics), f)
        
    ## Load the lists from the file
    with open(f'{params["main_dir"]}/PFstudy_results_k0is_{str(Best_w[0])}.pkl', 'rb') as f:
        load_ws, load_dec_train_loss, load_dec_val_accu, load_dec_val_orig_losses, load_dec_model_val_accu, load_dec_best_val_accu, load_dec_test_accu, load_dec_BEST_iter, load_dec_perc_wrong_pred, load_dec_alphas, load_dec_tr_metrics = pickle.load(f)

    # Bestw_TRAIN_LOSS = load_dec_train_loss[ind_best_w]
    # Bestw_VAL_ACCU = load_dec_val_accu[ind_best_w]

    # Bestw_MODEL_VAL_ACCU = load_dec_model_val_accu[ind_best_w]
    # Bestw_val_accu = load_dec_best_val_accu[ind_best_w]
    Bestw_test_accu = load_dec_test_accu[ind_best_w]
    # Bestw_orig_losses = load_dec_val_orig_losses[ind_best_w]

    # Bestw_perc_wrong_pred = load_dec_perc_wrong_pred[ind_best_w]
    # Bestw_BEST_iter = load_dec_BEST_iter[ind_best_w]
    # Bestw_dec_alphas = load_dec_alphas[ind_best_w]

    Bestw_dec_trMetrics= load_dec_tr_metrics[ind_best_w]

    print("Best k: ", Best_w)
    print("Test Accuracy = ", Bestw_test_accu.mean().item())
    print("Accuracy Task 1 = ", Bestw_test_accu[0].item())
    print("Accuracy Task 2 = ", Bestw_test_accu[1].item())

    print("####### Training Results ####### ")
    print("Sparsity Rate: ", Bestw_dec_trMetrics[1][0])
    print("Compression Rate: ", Bestw_dec_trMetrics[1][1])
    print("Parameter Sharing: ", Bestw_dec_trMetrics[1][2])
    print("################################ ")

    ##################################
    ############# PLOTS #############
    ##################################

    #Sparsity_mtrcs = [liste[1][0]/100 for liste in load_dec_tr_metrics]
    Pareto_front_loss = [liste[-1][-1][:,0].tolist() for liste in load_dec_val_orig_losses]
    Pareto_front_loss = np.array(Pareto_front_loss)

    PF_loss_3D = (Pareto_front_loss.T).tolist()
    x = PF_loss_3D[1]
    y = PF_loss_3D[2]
    z = PF_loss_3D[0]

    # best_otb_idx = np.argmax(np.array([tens.mean() for tens in load_dec_test_accu]))

    print("############ Results after epsilon-dominance test ############")
    epsilon = 0.00*np.min(Pareto_front_loss, axis = 0)
    print("epsilon = ", epsilon)
    PS_idx_a = eps_dominance(Pareto_front_loss, epsilon, start = 1)
    print(f"{len(PS_idx_a)} Pareto Optimal Points obtained (epsilon = {epsilon}) !")

    # ind_newbest_w = np.argmax(np.array([tens.mean().item() for tens in load_dec_test_accu])[PS_idx_a])
    # newBestw_dec_trMetrics= (np.array(load_dec_tr_metrics)[PS_idx_a]).tolist()[ind_newbest_w]
    # newBestw_test_accu= (np.array(load_dec_test_accu)[PS_idx_a]).tolist()[ind_newbest_w]
    # newBest_w = (np.array(ws)[PS_idx_a]).tolist()[ind_newbest_w]

    filtered_tr_metrics = [load_dec_tr_metrics[i] for i in PS_idx_a]
    filtered_test_accu = [load_dec_test_accu[i] for i in PS_idx_a]
    filtered_ws = [load_ws[i] for i in PS_idx_a]

    ind_newbest_w = np.argmax([acc.mean().item() for acc in filtered_test_accu])
    newBestw_dec_trMetrics = filtered_tr_metrics[ind_newbest_w]
    newBestw_test_accu = filtered_test_accu[ind_newbest_w]
    newBest_w = filtered_ws[ind_newbest_w]

    print("Best k: ", newBest_w)
    print("Test Accuracy = ", newBestw_test_accu.mean().item())
    print("Accuracy Task 1 = ", newBestw_test_accu[0].item())
    print("Accuracy Task 2 = ", newBestw_test_accu[1].item())

    print("Sparsity Rate: ", newBestw_dec_trMetrics[1][0])
    print("Compression Rate: ", newBestw_dec_trMetrics[1][1])
    print("Parameter Sharing: ", newBestw_dec_trMetrics[1][2])
    print("################################ ")

    Pareto_front_loss_filter = Pareto_front_loss[PS_idx_a]
    PF_loss_3D_filter = (Pareto_front_loss_filter.T).tolist()
    x_f = PF_loss_3D_filter[1]
    y_f = PF_loss_3D_filter[2]
    z_f = PF_loss_3D_filter[0]

    plt.figure()

    best_OP = [x_f[ind_newbest_w], y_f[ind_newbest_w], z_f[ind_newbest_w]]

    plt.scatter(best_OP[0], best_OP[1], c = "pink",
                s=100, facecolors='none', edgecolors='blue', alpha = 1,
                label="Best accuracy")
    # Plot the 2D scatter plot with X and Y axes
    plt.scatter(x_f, y_f, c = "red", alpha = 1)
        
    fsorted_indices1 = np.argsort(x_f)
    fcurve1 = np.array(x_f)[fsorted_indices1]
    fcurve2 = np.array(y_f)[fsorted_indices1]
    plt.plot(fcurve1, fcurve2, color="green", linestyle = "--")

    # Set labels for the axes
    plt.xlabel('Task 1')
    plt.ylabel('Task 2')
    plt.title(f'2D Pareto Front (Main objective functions)')
    plt.legend()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'Images/TwoDParetoFrontStudy_{archi_name}_k0is_{str(Best_w[0])}..png', dpi=300)

    # Show the plot
    plt.tight_layout()
    plt.show()
                    
    print("Pareto Front Study Completed !")

