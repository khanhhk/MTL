import torch
from tqdm import tqdm

def compute_accuracy(predictions, targets) -> float:
    return float((predictions.argmax(dim=1) == targets).sum().item() / len(predictions))

def train_multi(X, y, model, n_tasks, optimizer, loss_fn, batch_size, device):
    X = X.clone().detach().to(device).float()
    y = y.clone().detach().to(device).long()
    permutation = torch.randperm(X.size(0))
    losses = [[] for _ in range(n_tasks)]

    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X[indices], y[indices]

        if batch_x.size(0) != batch_size:
            continue

        model.zero_grad()

        # Get task-specific losses
        task_loss = model(batch_x, batch_y)

        # Compute gradients per task
        shared_gradients = [[] for _ in range(n_tasks)]
        for t in range(n_tasks):
            optimizer.zero_grad()
            task_loss[t].backward(retain_graph=True)
            for param in model.model.parameters():
                if param.grad is not None:
                    shared_gradients[t].append(param.grad.data.clone())

        alphas = optimizer.frank_wolfe_solver(shared_gradients, device=device)

        # Final backward pass with weighted sum
        loss = sum(alphas[t] * task_loss[t] for t in range(n_tasks))
        loss.backward()
        optimizer.step()

        for t in range(n_tasks):
            losses[t].append(task_loss[t].item())

    return losses

def test_multi(X, y, model, n_tasks, batch_size, device):
    X = X.clone().detach().to(device).float()
    y = y.clone().detach().to(device).long()
    model.eval()
    permutation = torch.randperm(X.size(0))
    task_accuracies = [[] for _ in range(n_tasks)]

    with torch.no_grad():
        for i in tqdm(range(0, X.size(0), batch_size)):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X[indices], y[indices]

            if batch_x.size(0) != batch_size:
                continue

            logits = model.model(batch_x)

            for t in range(n_tasks):
                t_output = logits[:, t, :]
                task_accuracies[t].append(compute_accuracy(t_output, batch_y[:, t]))

    return task_accuracies