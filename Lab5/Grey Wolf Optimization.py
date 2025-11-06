import numpy as np

def GWO(obj_func, dim, lb, ub, n_wolves=20, max_iter=100):
    lb, ub = np.array(lb), np.array(ub)
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))

    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    convergence_curve = []

    for iter in range(max_iter):
        for i in range(n_wolves):
            fitness = obj_func(wolves[i])
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = wolves[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = wolves[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        a = 2 - 2 * (iter / max_iter)

        for i in range(n_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3

            wolves[i] = np.clip(wolves[i], lb, ub)

        convergence_curve.append(alpha_score)
        print(f"Iteration {iter+1}/{max_iter}, Best Fitness = {alpha_score:.6f}")

    return alpha_pos, alpha_score, convergence_curve


if __name__ == "__main__":
    def sphere_function(x):
        return np.sum(x ** 2)

    dim = 2
    lb = [-5, -5]
    ub = [5, 5]
    n_wolves = 20
    max_iter = 50

    best_pos, best_score, curve = GWO(sphere_function, dim, lb, ub, n_wolves, max_iter)

    print("\nBest position found:", best_pos)
    print("Best objective value:", best_score)
