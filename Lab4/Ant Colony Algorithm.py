import random
import numpy as np

class AntColony:
    def __init__(self, graph, n_ants, n_iterations, alpha, beta, evaporation_rate, pheromone_init):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones_like(graph) * pheromone_init
        self.best_path = None
        self.best_path_length = float('inf')

    def _calculate_transition_probabilities(self, ant, visited):
        current_node = ant[-1]
        probabilities = []

        for j in range(len(self.graph)):
            if j not in visited and self.graph[current_node][j] > 0 and self.graph[current_node][j] != np.inf:
                pheromone = self.pheromone[current_node][j] ** self.alpha
                distance = (1.0 / self.graph[current_node][j]) ** self.beta
                probabilities.append(pheromone * distance)
            else:
                probabilities.append(0)

        total_pheromone = sum(probabilities)
        if total_pheromone == 0:
            return [0 for _ in probabilities]

        probabilities = [p / total_pheromone for p in probabilities]
        return probabilities

    def _construct_path(self, start_node, end_node):
        visited = set([start_node])
        path = [start_node]
        total_distance = 0

        while path[-1] != end_node:
            current_node = path[-1]
            probabilities = self._calculate_transition_probabilities(path, visited)

            if sum(probabilities) == 0:
                return None, float('inf')

            next_node = self._select_next_node(probabilities)
            visited.add(next_node)
            path.append(next_node)
            total_distance += self.graph[current_node][next_node]

        return path, total_distance

    def _select_next_node(self, probabilities):
        return np.random.choice(len(probabilities), p=probabilities)

    def _update_pheromones(self, paths, path_lengths):
        self.pheromone *= (1 - self.evaporation_rate)

        for path, length in zip(paths, path_lengths):
            pheromone_deposit = 1.0 / length
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_deposit

    def run(self, start_node, end_node):
        for iteration in range(self.n_iterations):
            paths = []
            path_lengths = []
            for _ in range(self.n_ants):
                path, length = self._construct_path(start_node, end_node)
                if path is not None:
                    paths.append(path)
                    path_lengths.append(length)

                    if length < self.best_path_length:
                        self.best_path_length = length
                        self.best_path = path

            self._update_pheromones(paths, path_lengths)
            print(f"Iteration {iteration + 1}: Best Path Length = {self.best_path_length}")

        return self.best_path, self.best_path_length

graph = np.array([
    [0, 1, 1, np.inf, np.inf],
    [1, 0, 1, 1, np.inf],
    [1, 1, 0, 1, 1],
    [np.inf, 1, 1, 0, 1],
    [np.inf, np.inf, 1, 1, 0]
])

n_ants = 10
n_iterations = 100
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
pheromone_init = 0.1

aco = AntColony(graph, n_ants, n_iterations, alpha, beta, evaporation_rate, pheromone_init)

best_path, best_path_length = aco.run(start_node=0, end_node=4)

print(f"Best Path: {best_path}")
print(f"Best Path Length: {best_path_length}")
