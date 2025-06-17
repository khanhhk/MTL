import pickle
import torch
import os
import gdown
import numpy as np 
import matplotlib.pyplot as plt
from model.lenet import RegressionModel, RegressionTrain

# Data
file_id = '1b4ZjhHC8zSeAjlsaCOu1j6ZMC7G3V9dU'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'multi_mnist.pickle'
if not os.path.exists(output):
    print("Downloading dataset...")
    gdown.download(url, output, quiet=False)
else:
    print(f"Dataset already exists at '{output}', skipping download.")

with open('multi_mnist.pickle','rb') as f:
    trainX, trainLabel,testX, testLabel = pickle.load(f)  

trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
trainLabel = torch.from_numpy(trainLabel).long()
testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()
testLabel = torch.from_numpy(testLabel).long()
train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
test_set  = torch.utils.data.TensorDataset(testX, testLabel) 
 
batch_size = 256
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# Model
state_dict = torch.load('logs/model_mtl.pickle', map_location='cpu')
n_tasks = 2
init_weight = np.array([0.5 , 0.5 ])
model = RegressionModel(n_tasks)
model.load_state_dict(state_dict)
model = RegressionTrain(model, init_weight)
if torch.cuda.is_available():
    model.cuda()

# Predict
X_batch, ts_batch = next(iter(test_loader))
if torch.cuda.is_available():
    X_batch = X_batch.cuda()
    ts_batch = ts_batch.cuda()

output = model.model(X_batch)
output1 = output.max(2, keepdim=True)[1][:, 0] 
output2 = output.max(2, keepdim=True)[1][:, 1]  

X_batch = X_batch.cpu()
ts_batch = ts_batch.cpu()
output1 = output1.cpu()
output2 = output2.cpu()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i >= 10:
        break
    img = X_batch[i].squeeze() 
    
    ax.imshow(img, cmap='gray')
    ax.axis("off")
    
    true_label = f"{ts_batch[i,0].item()} {ts_batch[i,1].item()}"
    pred_label = f"{output1[i].item()} {output2[i].item()}"
    ax.set_title(f"GT: {true_label} | Pred: {pred_label}")

plt.tight_layout()
plt.show()
