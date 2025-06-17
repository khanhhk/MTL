import torch
from config import get_params
from train_and_test_model_CM import load_Cifar10Mnist_data
import matplotlib.pyplot as plt

# ----- Thông số giống như khi train -----
k = [1e-2, 0.09, 0.9]
main_dir = "logs/MDMTN_CM_logs"
mod_logdir = "MDMTN_model_CM_onek"
archi_name = "MDMTN"
data_name = "Cifar10Mnist"
num_model = 0
if k[0] == 0:
    Sparsity_study = False
else:
    Sparsity_study = True

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Load tham số và mô hình -----
model, Cifar10mnist_params, GrOWL_parameters = get_params(k, archi_name, data_name, main_dir, mod_logdir, num_model, Sparsity_study)


# ----- Load trọng số mô hình đã huấn luyện -----
model_path = f"{main_dir}/{mod_logdir}/model000.pth"
model = torch.load(model_path, map_location=device, weights_only=False)
print("Loaded trained model from:", model_path)
model.to(device)
model.eval()

# ----- Load dữ liệu inference -----
_, _, test_loader = load_Cifar10Mnist_data()

# ----- Inference -----
data, target = next(iter(test_loader))
data = data.to(device)
target = [t.to(device) for t in target]
target = torch.stack(target, dim=1)
with torch.no_grad():
    outputs = model(data) 
output1 = torch.argmax(outputs[0], dim=1).cpu()
output2 = torch.argmax(outputs[1], dim=1).cpu()
data = data.cpu()
target = target.cpu()

def unnormalize(img_tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return img_tensor * std + mean

cifar10_classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
fig, axes = plt.subplots(3, 3, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i >= len(data):
        break
    img = unnormalize(data[i])
    img = torch.clamp(img, 0, 1)  # Đảm bảo ảnh hợp lệ
    img = img.permute(1, 2, 0).numpy()
    ax.imshow(img, cmap='gray')
    ax.axis("off")

    gt_label = f"{cifar10_classes[target[i][0].item()]} {target[i][1].item()}"
    pred_label = f"{cifar10_classes[output1[i].item()]} {output2[i].item()}"
    # ax.set_title(f"GT: {gt_label} | Pred: {pred_label}")
    ax.set_title(f"GT: {gt_label} | Pred: {pred_label}", fontsize=20)
plt.tight_layout()
plt.show()

