import random

def profit_function(x):
    return -0.5 * (x**2) + 20 * x - 30

class Particle:
    def __init__(self, lower_bound, upper_bound):
        self.position = random.uniform(lower_bound, upper_bound)
        self.velocity = random.uniform(-1, 1)
        self.pbest_position = self.position
        self.pbest_value = profit_function(self.position)

class PSO:
    def __call__(self, num_particles=10, max_iterations=20, lower_bound=0, upper_bound=50, w=0.5, c1=1.5, c2=1.5):
        swarm = [Particle(lower_bound, upper_bound) for _ in range(num_particles)]
        gbest_position = swarm[0].position
        gbest_value = swarm[0].pbest_value

        for particle in swarm:
            if particle.pbest_value > gbest_value:
                gbest_value = particle.pbest_value
                gbest_position = particle.pbest_position

        for iteration in range(max_iterations):
            for particle in swarm:
                profit = profit_function(particle.position)
                if profit > particle.pbest_value:
                    particle.pbest_value = profit
                    particle.pbest_position = particle.position
                if profit > gbest_value:
                    gbest_value = profit
                    gbest_position = particle.position

            for particle in swarm:
                r1 = random.random()
                r2 = random.random()
                particle.velocity = (
                    w * particle.velocity
                    + c1 * r1 * (particle.pbest_position - particle.position)
                    + c2 * r2 * (gbest_position - particle.position)
                )
                particle.position = particle.position + particle.velocity
                if particle.position < lower_bound:
                    particle.position = lower_bound
                if particle.position > upper_bound:
                    particle.position = upper_bound

            print(f"Iteration {iteration+1}: Best Workforce Size = {gbest_position:.2f}, Profit = {gbest_value:.2f}")

        return gbest_position, gbest_value

if __name__ == "__main__":
    best_pos, best_val = PSO()()
    print("\nFinal Result:")
    print(f"Optimal Workforce Size: {best_pos:.2f}, Maximum Profit: {best_val:.2f}")

     
