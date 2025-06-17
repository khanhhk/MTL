import torch
import torch.optim as optim

from src.utils.projectedOWL_utils import proxOWL

from src.MDMTN_model_MM import SparseMonitoredMultiTaskNetwork_I
from src.MDMTN_MGDA_model_MM import MDMTNmgda_MultiTaskNetwork_I
from src.MDMTN_model_CM import SparseMonitoredMultiTaskNetwork_II
from src.MDMTN_MGDA_model_CM import MDMTNmgda_MultiTaskNetwork_II
##################################################
#####  Helper function to get the parameters #####
##################################################

def get_params(k, archi_name, data_name, main_dir, mod_logdir, num_model, Sparsity_study = True):
    '''
        #### Parameters used:
        - `w: (list)` - Preference vector k.
        - `a: (list)` - Reference point.
        - `epsilon: (real: 0-1)` - Augmentation term coefficient in the modified Weighted Chebyshev scalarization method.
        - `num_tasks: (int)` - Number of tasks considered.
        - `max_iter: (int)` - Number of iterations by default.
        - `max_iter_search: (int)` - Number of iterations during the first phase of the training algorithm.
        - `max_iter_retrain: (int)` - Number of iterations during the second phase of the training algorithm.
        - `num_epochs: (int)` - Number of epochs per iteration.
        - `tol_epochs: (int)` - Maximum number of epochs to wait if there is no improvement in model performance.
        - `lr: (real)` - Initialization of the learning rate.
        - `LR_scheduler: (True/False)` - Whether to reduce the learning rate after a certain period.
        - `lr_sched_coef: (real)` - Reduction coefficient of the learning rate.
        - `mu: (real)` - Lagrangian multiplier μ.
        - `rho: (real)` - Coefficient for updating the Lagrangian multiplier μ.
        - `min_sparsRate: (%)` - Minimum sparsity rate for a model to be saved.
        - `max_layerSRate: (real: 0-1)` - Maximum sparsity rate for a layer.
        - `sim_preference: (real: 0-1)` - Similarity preference in the Affinity Propagation method.
        - `skip_layer: (int)` - The layers that have this number of neurons are skipped when applying GrOWL (preferably the input layer).
        - `is_search: (True/False)` - True: for the first training phase (searching for sparse model); False: for the second phase (Forcing parameter sharing).
        - `Sparsity_study: (True/False)` - True: for sparsity study; False: for Pareto front study
        - `base_optimizer: (torch.optim)` - Optimizer.
        - `criterion: (torch.nn.functional)` - Criterion.
        - `num_batchEpoch: (int)` - Number of batches to use for an epoch.
        - `num_model: (int)` - Model number.
        - `main_dir: (directory)` - Main directory for saving the training results.
        - `mod_logdir: (directory)` - Directory for saving the model.
    '''
    
    if data_name == "Cifar10Mnist":
        mod_params = {"w": k, "a": torch.zeros(3),
                        "epsilon": 0.0001, "num_tasks": 3, "num_outs": [10, 10],
                        "max_iter": 12, "max_iter_search": 3, "max_iter_retrain": 10,  
                        "lr": 0.0001, "lr_sched_coef": 0.98, "LR_scheduler": True, 
                    "num_epochs": 3, "num_epochs_search": 3, "num_epochs_retrain": 3, "tol_epochs": None,
                    "num_model": num_model,"main_dir": main_dir, "mod_logdir": mod_logdir,
                    "mu": 6.8e-08,
                    "rho": 2,
                    "base_optimizer": optim.Adam, "is_search": Sparsity_study, "Sparsity_study": Sparsity_study,
                    "criterion": torch.nn.functional.nll_loss,}
        
        GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                "proxOWL": proxOWL,
                "skip_layer": 3, # Skip layer with "3" neurons
                "sim_preference": 0.8,
                }
        

        if archi_name.lower() == "mdmtn":
            mod_params["min_sparsRate"] = 10.00 # (10 %)
            GrOWL_parameters["max_layerSRate"] = 0.3 # (30 %)
            model = SparseMonitoredMultiTaskNetwork_II(GrOWL_parameters, mod_params["num_outs"], static_a = [False, None])

        else: raise ValueError(f"Unknown model architecture {archi_name} !")

    elif data_name == "MultiMnist":
        mod_params = {"w": k, "a": torch.zeros(3),
                        "epsilon": 0.0001, "num_tasks": 3, "num_outs": [10, 10],
                        "max_iter": 12, "max_iter_search": 3, "max_iter_retrain": 10, 
                        "lr": 0.0025, "lr_sched_coef": 0.5, "LR_scheduler": True, 
                    "num_epochs": 3, "num_epochs_search": 3, "num_epochs_retrain": 3, "tol_epochs": None,
                    "num_model": num_model,"main_dir": main_dir, "mod_logdir": mod_logdir,
                    "mu": 2.5e-08,  
                    "rho": 2, 
                    "base_optimizer": optim.Adam, "is_search": Sparsity_study, "Sparsity_study": Sparsity_study,
                    "criterion": torch.nn.functional.nll_loss,}
        
        GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                "proxOWL": proxOWL,
                "skip_layer": 1, # Skip layer with "1" neuron
                    "sim_preference": 0.7, 
                }
    

        if archi_name.lower() == "mdmtn":
            mod_params["min_sparsRate"] = 20.00 # (20 %)
            GrOWL_parameters["max_layerSRate"] = 0.8 # (80 %)
            model = SparseMonitoredMultiTaskNetwork_I(GrOWL_parameters, mod_params["num_outs"], static_a = [False, None])

        else: raise ValueError(f"Unknown model architecture {archi_name} !")
    
    else: raise ValueError(f"Unknown dataset {data_name} !")

    return model, mod_params, GrOWL_parameters

##########################################################################

def get_params_mgda(archi_name, data_name, model_dir_path, device):

    if data_name == "Cifar10Mnist":
        mod_params_mgda = { "lr": 1e-2, "momentum": 0.9,
                     "model_repetitions": 5, "training_epochs": 100,
                     "archi": archi_name,"img_shp": (32, 32, 3), "model_dir_path": model_dir_path,
                     "batch_size": 256}
        
        if archi_name.lower() == "mdmtn":
            model = MDMTNmgda_MultiTaskNetwork_II(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
        else: raise ValueError(f"Unknown model architecture {archi_name} !")

    elif data_name == "MultiMnist":
        mod_params_mgda = { "lr": 1e-2, "momentum": 0.9,
                     "model_repetitions": 10, "training_epochs": 100,
                     "archi": archi_name,"img_shp": (28, 28, 1), "model_dir_path": model_dir_path,
                     "batch_size": 256}
        
        if archi_name.lower() == "mdmtn":
            model = MDMTNmgda_MultiTaskNetwork_I(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
        else: raise ValueError(f"Unknown model architecture {archi_name} !")

    else: raise ValueError(f"Unknown dataset {data_name} !")

    return model, mod_params_mgda

        
    
