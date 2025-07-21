import pickle
import torch


def load_MultiMnist_mgda(output_path="multi_mnist.pickle", batch_size=(256, 256)):
    with open(output_path, 'rb') as f:
        trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
    testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()

    trainLabel = torch.from_numpy(trainLabel).long()
    testLabel = torch.from_numpy(testLabel).long()

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set  = torch.utils.data.TensorDataset(testX, testLabel)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size[0],
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size[1],
        shuffle=False
    )

    return trainX, testX, trainLabel, testLabel, train_loader, test_loader