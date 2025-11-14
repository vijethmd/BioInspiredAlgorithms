import numpy as np
import math
import matplotlib.pyplot as plt

def levy_flight(beta, dim):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma_u
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1.0 / beta))

def simple_bounds(s, lb, ub):
    return np.clip(s, lb, ub)

def cuckoo_search(obj_func, lb, ub, n_nests=25, pa=0.25, n_iter=200,
                  beta=1.5, alpha=0.01, verbose=False, seed=None):

    if seed is not None:
        np.random.seed(seed)

    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    if lb.size == 1:
        lb = np.full((1,), lb.item())
    if ub.size == 1:
        ub = np.full(lb.shape, ub.item())
    d = lb.size

    nests = np.random.rand(n_nests, d) * (ub - lb) + lb
    fitness = np.array([obj_func(nests[i]) for i in range(n_nests)])

    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_fit = fitness[best_idx]
    history = [best_fit]

    for t in range(n_iter):
        for i in range(n_nests):
            step = levy_flight(beta, d)
            new_solution = nests[i] + alpha * step * (nests[i] - best)
            new_solution = simple_bounds(new_solution, lb, ub)
            new_fit = obj_func(new_solution)

            if new_fit < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fit
                if new_fit < best_fit:
                    best_fit = new_fit
                    best = new_solution.copy()

        n_abandon = int(pa * n_nests)
        if n_abandon > 0:
            worst_idx = np.argsort(fitness)[-n_abandon:]
            for idx in worst_idx:
                nests[idx] = np.random.rand(d) * (ub - lb) + lb
                fitness[idx] = obj_func(nests[idx])
                if fitness[idx] < best_fit:
                    best_fit = fitness[idx]
                    best = nests[idx].copy()

        history.append(best_fit)

        if verbose and (t % max(1, (n_iter // 10)) == 0 or t == n_iter-1):
            print(f"Iter {t+1}/{n_iter} — Best fitness: {best_fit:.6e}")

    return best, best_fit, history

def sphere(x):
    return np.sum(x**2)

if __name__ == "__main__":
    dim = 10
    lower = -5.12
    upper = 5.12
    best_sol, best_val, hist = cuckoo_search(
        obj_func=sphere,
        lb=np.full(dim, lower),
        ub=np.full(dim, upper),
        n_nests=40,
        pa=0.25,
        n_iter=400,
        beta=1.5,
        alpha=0.01,
        verbose=True,
        seed=42
    )

    print("\nBest solution found:")
    print(best_sol)
    print("Best value:", best_val)

    plt.figure(figsize=(8, 4))
    plt.plot(hist, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (log scale)")
    plt.title("Cuckoo Search convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
