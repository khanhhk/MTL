import torch
from torch import Tensor


def build_MGDA_optimizer(optimizer_type: torch.optim.Optimizer):
    """
    Builds a MGDAoptimizer Class derived from optimizer_type, e.g Adam or SGD.

    :param optimizer_type: class of the optimizer to use, e.g torch.optim.Adam
    :return: MGDAoptimizer: class
    """
    class MGDAoptimizer(optimizer_type):
        underlying_optimizer = optimizer_type

        @staticmethod
        def frank_wolfe_solver(gradients: list,
                               termination_threshold: float = 1e-4,
                               max_iterations: int = 10,
                               device: str = "cpu") -> Tensor:
            """
            Applies Frank-Wolfe-Solver to a list of (shared) gradients on t-many tasks
            :param gradients: list of (shared) gradients
            :param termination_threshold: termination condition
            :param max_iterations: #iterations before algorithm termination
            :return: Tensor of shape [t]
            """

            # Amount of tasks
            T = len(gradients)
            # Amount of layers
            L = len(gradients[0])

            # Initialize alpha
            alpha = torch.tensor([1 / T for _ in range(T)], device=device)

            M = torch.zeros(size=(T, T), dtype=torch.float32, device=device)


            for i in range(T):
                flat_gradient_i = torch.concat([torch.flatten(gradients[i][layer]) for layer in range(L)])
                for j in range(T):
                    flat_gradient_j = torch.concat([torch.flatten(gradients[j][layer]) for layer in range(L)])
                    if M[j][i] != 0:
                        M[i][j] = M[j][i]
                    else:
                        M[i][j] = torch.dot(flat_gradient_i, flat_gradient_j)

            # Initialize gamma
            gamma = float('inf')
            iteration = 0

            while gamma > termination_threshold and iteration <= max_iterations:
                alpha_m_sum = torch.matmul(alpha, M)
                t_hat = torch.argmin(alpha_m_sum)

                g_1 = torch.zeros_like(alpha, device=device)
                g_2 = alpha

                g_1[t_hat] = 1

                g1_Mg1 = torch.matmul((g_1), torch.matmul(M, g_1))
                g2_Mg2 = torch.matmul((g_2), torch.matmul(M, g_2))
                g1_Mg2 = torch.matmul((g_1), torch.matmul(M, g_2))

                if g1_Mg1 <= g1_Mg2:
                    gamma = 1
                elif g1_Mg2 >= g2_Mg2:
                    gamma = 0
                else:
                    dir_a = g2_Mg2 - g1_Mg2
                    dir_b = g1_Mg1 - 2*g1_Mg2 + g2_Mg2
                    gamma = dir_a / dir_b

                alpha = (1 - gamma) * alpha + gamma * g_1
                iteration += 1

                if T <= 2:
                    break
            return alpha

    return MGDAoptimizer
