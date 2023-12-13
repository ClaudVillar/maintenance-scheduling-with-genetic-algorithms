import numpy as np

# Define the problem parameters
num_units = 7
num_intervals = 4
max_loads = [80, 90, 65, 70]
totalcap = 150  # Initialize totalcap

# Define the GA parameters
pop_size = 10
num_generations = 20
mutation_rate = 0.01
tournament_size = 3

# Define the unit data for its capacities and maintenance intervals
unitData = np.array([
    (20, 2),
    (15, 2),
    (35, 1),
    (40, 1),
    (15, 1),
    (15, 1),
    (10, 1)
])

# Generate a chromosome
def generateGenome(unitData, num_intervals):
    # Generate an empty chromosome
    chromosome = np.zeros((num_units, num_intervals), dtype=int)

    for unit in range(num_units):
        if unitData[unit][1] > num_intervals:
            raise ValueError("Unit's required interval must not be greater than the load interval")

        # Maintenance of any unit starts at the beginning of an interval and finishes at the same or adjacent interval.
        startInterval = np.random.randint(0, num_intervals - unitData[unit][1] + 1)

        for interval in range(startInterval, startInterval + unitData[unit][1]):
            chromosome[unit, interval] = 1

    return chromosome

def roulette_wheel_selection(population):
    fitness_values = np.array([individual["fitness"] for individual in population])
    selection_probabilities = fitness_values / fitness_values.sum()

    selected_index = np.random.choice(len(population), p=selection_probabilities)
    return population[selected_index]["chromosome"]

def crossover(parent1, parent2, crossover_rate):
    # Check if crossover should be performed based on the crossover rate
    if np.random.rand() < crossover_rate:
        # Perform crossover
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    else:
        # If no crossover, children are identical to parents
        child1, child2 = parent1.copy(), parent2.copy()

    return child1, child2

def mutate(chromosome, mutation_rate):
    # Flip bits with probability determined by the mutation rate
    mutated_chromosome = chromosome ^ (np.random.rand(*chromosome.shape) < mutation_rate).astype(int)

    return mutated_chromosome

def calculate_fitness(chromosome, unitData, num_intervals, max_loads, totalcap):
    net_reserves = np.zeros(num_intervals, dtype=int)
    numChromosome = len(chromosome)

    for col in range(num_intervals):
        totalLoad = np.sum(chromosome[:, col] * unitData[:, 0])
        net_reserves[col] = totalcap - totalLoad - max_loads[col]

    # If the net reserve at any interval is negative, the schedule is illegal, and the fitness function returns zero
    if any(net_reserves < 0):
        return 0

    # Define the fitness value as the lowest net reserve    
    return np.min(net_reserves)

def initialize_population(pop_size, unitData, num_intervals, max_loads, totalcap):
    population = []
    for _ in range(pop_size):
        chromosome = generateGenome(unitData, num_intervals)
        fitness_value = calculate_fitness(chromosome, unitData, num_intervals, max_loads, totalcap)
        population.append({"chromosome": chromosome, "fitness": fitness_value})
    return population

def genetic_algorithm(unitData, num_intervals, max_loads, totalcap, pop_size, num_generations, mutation_rate, tournament_size):
    # Initialize population
    population = initialize_population(pop_size, unitData, num_intervals, max_loads, totalcap)

    for generation in range(num_generations):
        # Select parents using tournament selection
        parents = [roulette_wheel_selection(population) for _ in range(2)]

        # Perform crossover to generate offspring
        offspring1, offspring2 = crossover(parents[0], parents[1], crossover_rate=0.8)

        # Perform mutation on offspring
        offspring1 = mutate(offspring1, mutation_rate)
        offspring2 = mutate(offspring2, mutation_rate)

        # Calculate fitness for offspring
        fitness_offspring1 = calculate_fitness(offspring1, unitData, num_intervals, max_loads, totalcap)
        fitness_offspring2 = calculate_fitness(offspring2, unitData, num_intervals, max_loads, totalcap)

        # Replace the least fit individuals in the population with the offspring
        population.sort(key=lambda x: x["fitness"])
        population[0] = {"chromosome": offspring1, "fitness": fitness_offspring1}
        population[1] = {"chromosome": offspring2, "fitness": fitness_offspring2}

        # Print the best individual in the current generation
        best_individual = max(population, key=lambda x: x["fitness"])
        print(f"Generation {generation + 1}, Best Fitness: {best_individual['fitness']}")

    # Return the final population
    return population

# Run the genetic algorithm
final_population = genetic_algorithm(unitData, num_intervals, max_loads, totalcap, pop_size, num_generations, mutation_rate, tournament_size)

# Print the best individual in the final population
best_individual = max(final_population, key=lambda x: x["fitness"])
print("\nBest Individual in the Final Population:")
print("Chromosome:")
print(best_individual["chromosome"])
print("Fitness:", best_individual["fitness"])
