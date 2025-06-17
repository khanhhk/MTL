import torch
from config import get_params
from train_and_test_model_MM import load_MultiMnist_data
import matplotlib.pyplot as plt

# ----- Thông số giống như khi train -----
k = [0.001, 0.2, 0.799]
main_dir = "logs/MDMTN_MM_logs"
mod_logdir = "MDMTN_model_MM_onek"
archi_name = "MDMTN"
data_name = "MultiMnist"
num_model = 0
if k[0] == 0:
    Sparsity_study = False
else:
    Sparsity_study = True

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Load tham số và mô hình -----
model, Multimnist_params, GrOWL_parameters = get_params(k, archi_name, data_name, main_dir, mod_logdir, num_model, Sparsity_study)


# ----- Load trọng số mô hình đã huấn luyện -----
model_path = f"{main_dir}/{mod_logdir}/model000.pth"
model = torch.load(model_path, map_location=device, weights_only=False)
print("Loaded trained model from:", model_path)
model.to(device)
model.eval()

# ----- Load dữ liệu inference -----
_, _, test_loader = load_MultiMnist_data()

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
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i >= len(data):
        break

    img = data[i].squeeze()
    ax.imshow(img, cmap='gray')
    ax.axis("off")

    gt_label = f"{target[i][0].item()} {target[i][1].item()}"
    pred_label = f"{output1[i].item()} {output2[i].item()}"
    # ax.set_title(f"GT: {gt_label} | Pred: {pred_label}")
    ax.set_title(f"GT: {gt_label} | Pred: {pred_label}", fontsize=16)
# plt.tight_layout()
plt.show()

