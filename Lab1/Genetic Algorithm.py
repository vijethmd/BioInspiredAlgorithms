import random

CHROME_LENGTH = 5
POP_SIZE = 4
CROSS_RATE = 0.8
MUT_RATE = 0.1

def fitness(x):
  return x**2

def encode(x):
  if x >= 0:
    return format(x, f'0{CHROME_LENGTH}b')
  else:
    return bin((1 << CHROME_LENGTH) + x)[2:]


def decode(x):
  if x[0] == '0':
    return int(x, 2)
  else:
    return int(x, 2) - (1 << CHROME_LENGTH)

pop = []
for i in range(POP_SIZE):
  pop.append(random.randint(-10, 10))

#print(INITIAL_POPULATION)

def roulette_selection(pop,fitness):
  total_fitness = sum(fitness)
  pick = random.uniform(0,total_fitness)
  current = 0
  for i in range(len(pop)):
    current += fitness[i]
    if current > pick:
      return pop[i]
  return pop[-1]

def crossover(parent1,parent2):
  if random.random() < CROSS_RATE:
    index = random.randint(1,CHROME_LENGTH-1)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]
    return child1,child2
  else:
    return parent1,parent2

def mutate(child):
  child = list(child)
  for i in range(len(child)):
    if random.random() < MUT_RATE:
      index = random.randint(0,CHROME_LENGTH-1)
      child[index] = '1' if child[index] == '0' else '0'
  return ''.join(child)

def genetic_algorithm():

    population = [encode(x) for x in pop]
    print("Initial Population:", population, [decode(c) for c in population])

    for gen in range(1, 11):

        decoded = [decode(c) for c in population]
        fitnesses = [fitness(x) for x in decoded]

        total_fit = sum(fitnesses)
        probs = [f/total_fit for f in fitnesses]
        expected = [p*POP_SIZE for p in probs]

        print(f"\nGeneration {gen}")
        for i in range(POP_SIZE):
            print(f"x={decoded[i]}, bin={population[i]}, fit={fitnesses[i]}, "
                  f"prob={probs[i]:.3f}, exp_count={expected[i]:.2f}")

        new_pop = []
        while len(new_pop) < POP_SIZE:

            p1 = roulette_selection(population, fitnesses)
            p2 = roulette_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1, c2 = mutate(c1), mutate(c2)
            new_pop.extend([c1, c2])

        population = new_pop[:POP_SIZE]

    decoded = [decode(c) for c in population]
    fitnesses = [fitness(x) for x in decoded]
    best_idx = fitnesses.index(max(fitnesses))
    print("\nFinal Best Solution:", decoded[best_idx], population[best_idx], "fitness=", fitnesses[best_idx])

genetic_algorithm()
