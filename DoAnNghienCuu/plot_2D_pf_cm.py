import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.utils.PFstudy_utils import eps_dominance

# === CONFIG ===
main_dir = "logs/MDMTN_CM_logs/Pareto_Front_Study"
archi_name = "MDMTN"
k0 = 0.01
ws_file = f"{main_dir}/PFstudy_results_k0is_{str(k0)}.pkl"

# === LOAD ===
with open(f'{main_dir}/PFstudy_results_k0is_{str(k0)}.pkl', 'rb') as f:
    load_ws, load_dec_train_loss, load_dec_val_accu, load_dec_val_orig_losses, load_dec_model_val_accu, load_dec_best_val_accu, load_dec_test_accu, load_dec_BEST_iter, load_dec_perc_wrong_pred, load_dec_alphas, load_dec_tr_metrics = pickle.load(f)

# === PARETO LOSS ===
Pareto_front_loss = [liste[-1][-1][:,0].tolist() for liste in load_dec_val_orig_losses]
Pareto_front_loss = np.array(Pareto_front_loss)

# === EPSILON-DOMINANCE ===
epsilon = 0.00 * np.min(Pareto_front_loss, axis=0)
PS_idx_a = eps_dominance(Pareto_front_loss, epsilon, start=1)

# === CHỌN BEST TRONG PARETO SET ===
filtered_tr_metrics = [load_dec_tr_metrics[i] for i in PS_idx_a]
filtered_test_accu = [load_dec_test_accu[i] for i in PS_idx_a]
filtered_ws = [load_ws[i] for i in PS_idx_a]

ind_newbest_w = np.argmax([acc.mean().item() for acc in filtered_test_accu])
newBestw_dec_trMetrics = filtered_tr_metrics[ind_newbest_w]
newBestw_test_accu = filtered_test_accu[ind_newbest_w]
newBest_w = filtered_ws[ind_newbest_w]

# === IN LOG ===
print("Best k: ", newBest_w)
print("Test Accuracy = ", newBestw_test_accu.mean().item())
print("Accuracy Task 1 = ", newBestw_test_accu[0].item())
print("Accuracy Task 2 = ", newBestw_test_accu[1].item())

print("Sparsity Rate: ", newBestw_dec_trMetrics[1][0])
print("Compression Rate: ", newBestw_dec_trMetrics[1][1])
print("Parameter Sharing: ", newBestw_dec_trMetrics[1][2])
print("################################ ")

# === VẼ PARETO FRONT 2D ===
Pareto_front_loss_filter = Pareto_front_loss[PS_idx_a]
PF_loss_3D_filter = (Pareto_front_loss_filter.T).tolist()
x_f = PF_loss_3D_filter[1]
y_f = PF_loss_3D_filter[2]
z_f = PF_loss_3D_filter[0]

plt.figure()

best_OP = [x_f[ind_newbest_w], y_f[ind_newbest_w], z_f[ind_newbest_w]]

plt.scatter(best_OP[0], best_OP[1], c = "pink",
            s=100, facecolors='none', edgecolors='blue', alpha = 1,
            label="Best accuracy")
# Plot the 2D scatter plot with X and Y axes
plt.scatter(x_f, y_f, c = "red", alpha = 1)
    
fsorted_indices1 = np.argsort(x_f)
fcurve1 = np.array(x_f)[fsorted_indices1]
fcurve2 = np.array(y_f)[fsorted_indices1]
plt.plot(fcurve1, fcurve2, color="green", linestyle = "--")

# Set labels for the axes
plt.xlabel('Task 1')
plt.ylabel('Task 2')
plt.title(f'Hình chiếu 2D của mặt Pareto')
plt.legend()
plt.subplots_adjust(wspace=0.3)
plt.savefig(f'Images/TwoDParetoFrontStudy_{archi_name}_k0is_{str(k0)}..png', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()
                
print("Pareto Front Study Completed !")