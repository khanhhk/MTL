from model.lenet import RegressionModel, RegressionTrain
from model.resnet import MnistResNet, RegressionTrainResNet

def get_params_mgda(base_model, n_tasks, init_weight, data_name, device):
    '''
        #### Parameters used:
        - `model_repetitions: (int)` - số lần lặp mô hình
    '''
    if data_name == "MultiMnist":
        params_mgda = { "lr": 1e-2, "momentum": 0.9,
                     "training_epochs": 100,
                     "batch_size": [256, 256]}
        
        if base_model.lower() == "lenet":
            model = model = RegressionTrain(RegressionModel(n_tasks), init_weight).to(device)
        elif base_model.lower() == "resnet18":
            model = RegressionTrainResNet(MnistResNet(n_tasks), init_weight).to(device)
        else: raise ValueError(f"Unknown base model {base_model} !")

    else: raise ValueError(f"Unknown dataset {data_name} !")

    return model, params_mgda