import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from train_and_test_model_MM import train_and_test_model_MM
from config import get_params

k = [0.001, 0.099, 0.9] # (k_0, k_1, k_2)
main_dir = "logs/MDMTN_MM_logs"
mod_logdir = "MDMTN_model_MM_onek"
archi_name = "MDMTN"
data_name = "MultiMnist"
num_model = 0
if k[0] == 0:
    Sparsity_study = False
else:
    Sparsity_study = True

if __name__ == "__main__":

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    model, Multimnist_params, GrOWL_parameters = get_params(k, archi_name, data_name, main_dir, mod_logdir, num_model, Sparsity_study)

    Multimnist_params["device"] = device

    Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = train_and_test_model_MM(model, Multimnist_params)

    import pickle

    ## Save the results lists to a file
    with open(f'logs/MDMTN_MM_logs/MultiMnist_results_k1_{k[0]}_k2_{k[1]}_k3_{k[2]}.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], [TR_metrics, Best_iter], ALL_TRAIN_LOSS, ALL_VAL_ACCU,
                     ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu), f)
        
