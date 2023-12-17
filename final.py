import matplotlib.pyplot as plt
import numpy as np

# Define the problem parameters
num_units = 7
num_intervals = 4
max_loads = [80, 90, 65, 70]
totalcap = 0  # Initialize totalcap

# Define the GA parameters
pop_size = 20
num_generations = 100
mutation_rate = 0.001
crossover_rate = 0.7

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

# Find the total installed capacity of the units
totalcap = sum(unitData[i][0] for i in range(len(unitData)))

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

# Fitness-Proportionate Selection
# where the fitness of an individual is directly proportional to its probability of being selected
def roulette_wheel_selection(population):
    fitness_values = np.array([individual["fitness"] for individual in population])
    selection_probabilities = fitness_values / fitness_values.sum()

    selected_index = np.random.choice(len(population), p=selection_probabilities)
    return population[selected_index]["chromosome"]

# Classic crossover function
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

# Classic mutation function
def mutate(chromosome, mutationRate):
    mutated = np.copy(chromosome)
    mutationMask = (np.random.rand(*chromosome.shape) < mutationRate).astype(int)

    for unit in range(len(unitData)):
        requiredMaintenance = unitData[unit][1]

        # Ensure that the mutation maintains the required maintenance intervals
        for i in range(num_intervals - requiredMaintenance + 1):
            if mutationMask[unit, i:i + requiredMaintenance].all():
                mutated[unit, i:i + requiredMaintenance] = 1

    return mutated

# Fitness function based on the net_reserves of the chromosome of an individual
def calculate_fitness(chromosome, unitData, num_intervals, max_loads, totalcap):
    net_reserves = np.zeros(num_intervals, dtype=int)
    numChromosome = len(chromosome)

    for col in range(num_intervals):
        totalLoad = 0

        # totalLoad is the sum of capacities of the units scheduled for maintenance at each interval
        for row in range(numChromosome):
            totalLoad += (chromosome[row, col] * unitData[row][0])
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

def genetic_algorithm(unitData, num_intervals, max_loads, totalcap, pop_size, num_generations, mutation_rate, crossover_rate):
    # Initialize population
    population = initialize_population(pop_size, unitData, num_intervals, max_loads, totalcap)

    # Lists to store the best and average fitness values in each generation
    best_fitness_values = []
    average_fitness_values = []

    for generation in range(num_generations):
        # Select parents using Fitness-Proportionate selection
        parents = [roulette_wheel_selection(population) for _ in range(2)]

        # Perform crossover to generate offspring
        offspring1, offspring2 = crossover(parents[0], parents[1], crossover_rate)

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

        # Store the best fitness value in the current generation
        best_individual = max(population, key=lambda x: x["fitness"])
        best_fitness_values.append(best_individual['fitness'])

        # Calculate and store the average fitness value in the current generation
        average_fitness = np.mean([individual["fitness"] for individual in population])
        average_fitness_values.append(average_fitness)

        print(f"Generation {generation + 1}, Best Fitness: {best_individual['fitness']}, Average Fitness: {average_fitness}")

    # Plotting the best and average fitness values in every generation
    plt.plot(range(1, num_generations + 1), best_fitness_values, label='Best Fitness')
    plt.plot(range(1, num_generations + 1), average_fitness_values, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Best and Average Fitness in Every Generation')
    plt.legend()
    plt.show()

    # Return the final population
    return population

def print_best_chromosome(final_population):
    best_individual = max(final_population, key=lambda x: x["fitness"])
    print("\nBest Chromosome in the Final Population:")
    print("Chromosome:")
    print(best_individual["chromosome"])
    print("Fitness:", best_individual["fitness"])



# Run the genetic algorithm
final_population = genetic_algorithm(unitData, num_intervals, max_loads, totalcap, pop_size, num_generations, mutation_rate, crossover_rate)

def plot_chromosome(chromosome, unitData, max_loads, totalcap):
    num_units, num_intervals = chromosome.shape
    x_values = np.arange(num_intervals) + 1  # Adjusted to start from 1 for intervals

    # Reshape unit capacities for correct alignment with the chromosome
    unit_capacities = unitData[:, 0].reshape(num_units, 1)

    # Calculate the cumulative sum of unit capacities for each interval
    cumulative_sum = np.cumsum(chromosome * unit_capacities, axis=0)

    # Plot the cumulative sum for each unit
    for unit in range(num_units):
        plt.bar(x_values, cumulative_sum[unit], bottom=np.sum(cumulative_sum[:unit], axis=0), label=f'Unit {unit + 1}')

    # Add annotations for max loads, total loads, and net reserves
    for col in range(num_intervals):
        total_load = np.sum(chromosome[:, col] * unit_capacities)
        net_reserve = totalcap - total_load - max_loads[col]

        plt.text(x_values[col], totalcap + 5, f'Max Load: {max_loads[col]}', ha='center')
        plt.text(x_values[col], totalcap + 15, f'Total Load: {total_load}', ha='center')
        plt.text(x_values[col], totalcap + 25, f'Net Reserve: {net_reserve}', ha='center')

    plt.xlabel('Intervals')
    plt.ylabel('Cumulative Capacity Used')
    plt.title('Chromosome Representation')
    plt.legend()
    plt.show()

# Plot the best chromosome
best_chromosome = max(final_population, key=lambda x: x["fitness"])["chromosome"]

# Print the best chromosome in the final population
print_best_chromosome(final_population)
plot_chromosome(best_chromosome, unitData, max_loads, totalcap)