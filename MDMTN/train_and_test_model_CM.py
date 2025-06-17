from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data.cifar10mnist_dataloader import Cifar10Mnist_loaders
from src.utils.WCsAL_Test import test_multitask_model
from src.utils.WCsAL_Train import full_training

def load_Cifar10Mnist_data():
    data_path = "Data/Cifar10Mnist"
    split_rate = 0.8
    batch_size = [256, 256] # (train, test)

    train_transform = transforms.Compose([transforms.Lambda(lambda x: Image.fromarray(np.uint8(x))), #
                                        transforms.RandomRotation(20), #
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.Lambda(lambda x: x.permute(0, 1, 2))
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.Lambda(lambda x: x.permute(0, 1, 2))
                                        ])

    transformers = [train_transform, test_transform] # [None, None]
    train_loader, val_loader, test_loader = Cifar10Mnist_loaders(data_path, split_rate, transformers, batch_size)
    print("Data loaded!")
    return train_loader, val_loader, test_loader 

def train_and_test_model_CM(model, Cifar10mnist_params):

    # Start timer
    import datetime
    from time import time
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_Cifar10Mnist_data()

    # # Choose device
    device = Cifar10mnist_params["device"]
    print(f"Training... [--- running on {device} ---]")
    
    final_model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = full_training(train_loader, val_loader, model,
                        Cifar10mnist_params, init_model = True)
    
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_model(test_loader, final_model, Cifar10mnist_params, TR_metrics) 

    return Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter


#################################################################################