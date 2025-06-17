import torch
from torchvision import transforms
from data.get_multimnist_dataset import get_dataset


def MultiMnist_loaders(data_path, split_rate, transformers = [None, None], batch_size = [256, 256], SEED = 42):

    configs = {
    "mnist": {
        "path": data_path,
        "all_tasks": ["L", "R"]
        },
        }
    
    params = {
        "optimizer": "Adam", 
        "batch_size": batch_size,
        "lr": 0.0001,
        "dataset": "mnist",
        "tasks": ["0", "1"],
        "scales": {"0":0.025, "1":0.025},
        "parallel": True
    }

    def Transform(transform):
        if transform is None:
            return transforms.Compose([transforms.ToTensor()])
        else:
            return transform
        
    transformers = [Transform(transformers[0]), Transform(transformers[1])]
    
    _, mm_train_dst, _, mm_test_dst = get_dataset(params, configs, transformers)

    # Split the train_loader into train and validation loaders
    if (split_rate <= 0) or (split_rate >= 1):
        raise ValueError(" Use 0<split_rate<1")
    generator = torch.Generator().manual_seed(SEED)
    train_size = int(split_rate * len(mm_train_dst))
    val_size = len(mm_train_dst) - train_size
    train_set, val_set = torch.utils.data.random_split(mm_train_dst, [train_size, val_size], generator = generator)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size[0], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size[1], shuffle=False)
    test_loader = torch.utils.data.DataLoader(mm_test_dst, batch_size=batch_size[1], shuffle=False)

    return train_loader, val_loader, test_loader