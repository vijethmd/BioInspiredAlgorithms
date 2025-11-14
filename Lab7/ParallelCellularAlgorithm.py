import numpy as np

def fitness(x):
    return x**2 - 4*x + 4

def neighborhood_average(grid):
    s = np.zeros_like(grid)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            s += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return s / 9.0

def parallel_cellular_minimize(fitness_fn, grid_shape=(10,10),
                               low=-10, high=10, iterations=100, seed=42):

    rng = np.random.RandomState(seed)
    grid = rng.uniform(low, high, size=grid_shape)

    for it in range(iterations):
        fit = fitness_fn(grid)
        min_idx = np.unravel_index(np.argmin(fit), fit.shape)
        min_val = fit[min_idx]
        min_x = grid[min_idx]

        print(f"Iteration {it+1}: best x = {min_x:.6f}, fitness = {min_val:.6f}")

        avg = neighborhood_average(grid)
        grid = avg.copy()

    final_fit = fitness_fn(grid)
    final_min_idx = np.unravel_index(np.argmin(final_fit), final_fit.shape)
    final_min_val = final_fit[final_min_idx]

    return {
        "final_best_index": final_min_idx,
        "final_best_value": final_min_val,
        "final_best_x": grid[final_min_idx]
    }

if __name__ == "__main__":
    result = parallel_cellular_minimize(fitness,
                                        grid_shape=(10,10),
                                        low=-10, high=10,
                                        iterations=100,
                                        seed=1)

    print("\nFinal Best Location:", result["final_best_index"])
    print("Final Best x:", result["final_best_x"])
    print("Final Fitness:", result["final_best_value"])
