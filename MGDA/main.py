import torch
import numpy as np
import os
import gdown
from train import train_test_MGDA

from config import get_params_mgda

base_model = "lenet"
n_tasks = 2
init_weight = np.array([0.5 , 0.5 ])
data_name = "MultiMnist"

if __name__ == "__main__":
    file_id = '1b4ZjhHC8zSeAjlsaCOu1j6ZMC7G3V9dU'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'multi_mnist.pickle'
    if not os.path.exists(output):
        print("Downloading dataset...")
        gdown.download(url, output, quiet=False)
    else:
        print(f"Dataset already exists at '{output}', skipping download.")

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    model, Multimnist_params_mgda = get_params_mgda(base_model, n_tasks, init_weight, data_name, device)

    train_losses, test_accuracies = train_test_MGDA(model, n_tasks, data_name, Multimnist_params_mgda, device)

    import pickle

    # Save the results lists to a file
    with open(f'logs/MultiMnist_results.pkl', 'wb') as f:
        pickle.dump((train_losses, test_accuracies), f)

    print("Done!")
