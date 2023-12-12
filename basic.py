import numpy as np

# Define the problem parameters
num_units = 7
num_intervals = 4
max_loads = [80, 90, 65, 70]

# Define the GA parameters
pop_size = 6
num_generations = 1000
mutation_rate = 0.001

# Define the unit capacities and maintenance intervals
unit_capacities = np.array([20, 15, 35, 40, 15, 15, 10])
maintenance_intervals = np.array([2, 2, 1, 1, 1, 1, 1])

# Define the chromosome representation
class Chromosome:
    def __init__(self):
        self.genes = np.random.randint(2, size=num_units)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        # Calculate the fitness of the chromosome
        # This function should be modified to fit the specific problem
        return np.sum(unit_capacities[self.genes == 0])

# Initialize the population
population = [Chromosome() for _ in range(pop_size)]

for generation in range(num_generations):
    # Selection
    parents = np.random.choice(population, size=2, p=[c.fitness/np.sum([c.fitness for c in population]) for c in population])

    # Crossover
    crossover_point = np.random.randint(num_units)
    child1 = Chromosome()
    child1.genes = np.concatenate((parents[0].genes[:crossover_point], parents[1].genes[crossover_point:]))
    child2 = Chromosome()
    child2.genes = np.concatenate((parents[1].genes[:crossover_point], parents[0].genes[crossover_point:]))

    # Mutation
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(num_units)
        child1.genes[mutation_point] = 1 - child1.genes[mutation_point]  # Flip the gene from 0 to 1 or 1 to 0
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(num_units)
        child2.genes[mutation_point] = 1 - child2.genes[mutation_point]  # Flip the gene from 0 to 1 or 1 to 0

    # Replacement
    population.remove(min(population, key=lambda c: c.fitness))
    population.remove(min(population, key=lambda c: c.fitness))
    population.append(child1)
    population.append(child2)

    # Print the best solution of each generation
    best_solution = max(population, key=lambda c: c.fitness)
    print(f"Generation {generation+1}: Best solution: {best_solution.genes}, Fitness: {best_solution.fitness}")

# The best solution is the one with the highest fitness
best_solution = max(population, key=lambda c: c.fitness)
print("Final Best solution:", best_solution.genes)
