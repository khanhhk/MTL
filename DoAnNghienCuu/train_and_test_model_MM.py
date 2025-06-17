from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from data.multimnist_dataloader import MultiMnist_loaders
from src.utils.WCsAL_Train import full_training
from src.utils.WCsAL_Test import test_multitask_model

def load_MultiMnist_data():
    data_path = "data"
    split_rate = 0.83334
    batch_size = [256, 100] # (train, test)

    train_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    transformers = [train_transform, test_transform] # [None, None]
    train_loader, val_loader, test_loader = MultiMnist_loaders(data_path, split_rate, transformers, batch_size)
    print("Data loaded!")

    # print("Show sample image...")
    # train_dataiter = iter(train_loader)
    # images, targets = next(train_dataiter)
    # img = images[0]
    # plt.figure(figsize=(5, 5))
    # img = np.transpose(img, (1, 2, 0))
    # plt.imshow(img)
    # plt.title(f"({targets[0][0].item()}, {targets[1][0].item()})")
    # plt.axis('off')
    # plt.show()

    return train_loader, val_loader, test_loader

def train_and_test_model_MM(model, MultiMNISt_params):

    # Start timer
    import datetime
    from time import time
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    device = MultiMNISt_params["device"]
    print(f"Training... [--- running on {device} ---]")
    
    final_model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = full_training(train_loader, val_loader, model,
                          MultiMNISt_params, init_model = True)
    
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_model(test_loader, final_model, MultiMNISt_params, TR_metrics) 

    return Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter