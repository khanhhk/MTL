import torch
from tqdm import tqdm

def compute_accuracy(predictions, targets) -> float:
    return float(
        sum([pred.argmax() == target.argmax() for pred, target in zip(predictions, targets)]) / len(predictions))

def train_multi(X, y, model, optimizer, loss_fn, batch_size, device, archi, img_shp = (32, 32, 3)):
    # X is a torch Variable
    permutation = torch.randperm(X.size()[0])
    losses = [[] for _ in range(model.num_of_tasks)]

    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X[indices].to(device), y[indices].to(device)
        # Only accept full-size batches
        if batch_x.shape[0] != batch_size:
            continue
        batch_x = batch_x.reshape([batch_size, 1, img_shp[0], img_shp[1], img_shp[2]])

        shared_gradients = [[] for _ in range(model.num_of_tasks)]
        model.zero_grad()
        for t in range(model.num_of_tasks):
            # Full forward pass
            output = model.forward(batch_x)

            # Compute t-specific loss
            t_output = output[:, t * model.task_out_n:(t + 1) * model.task_out_n]
            t_label = batch_y[:, t]
            t_loss = loss_fn(t_output, t_label)

            # Backward pass
            t_loss.backward()

            for param in model.shared.parameters():
                if param.grad is not None:
                    _grad = param.grad.data.detach().clone()
                    shared_gradients[t].append(_grad)

        alphas = optimizer.frank_wolfe_solver(shared_gradients, device=device)

        # Collect task specific gradients regarding task specific loss
        z = model.shared_forward(batch_x)

        # aggregate loss
        loss = torch.zeros(1, device=device)
        for t in range(model.num_of_tasks):
            if archi == "hps":
                loss_t = loss_fn(model.taskOutput[f"task_{t}"].forward(z), batch_y[:, t]) * alphas[t]
            elif archi == "mdmtn":
                loss_t = loss_fn(model.taskOutput[f"task_{t}"].forward(z[t]), batch_y[:, t]) * alphas[t]
            else: raise ValueError("Model Architecture should be 'hps' or 'mdmtn' !")
            loss += loss_t
            losses[t].append(loss_t.detach().item())

        loss.backward()

        optimizer.step()
    return losses

def test_multi(X, y, model, loss_fn, batch_size, device, img_shp = (32, 32, 3)):
    model.eval()
    permutation = torch.randperm(X.size()[0])
    task_accuracies = [[] for _ in range(model.num_of_tasks)]

    with torch.no_grad():
        for i in tqdm(range(0, X.size()[0], batch_size)):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X[indices].to(device), y[indices].to(device)
            if batch_x.shape[0] != batch_size:
                continue
            batch_x = batch_x.reshape([batch_size, 1, img_shp[0], img_shp[1], img_shp[2]])
            pred = model(batch_x)

            for t in range(model.num_of_tasks):
                t_output = pred[:, t * model.task_out_n:(t + 1) * model.task_out_n]
                # Compute classification accuracy per task
                task_accuracies[t].append(compute_accuracy(t_output, batch_y[:, t]))
