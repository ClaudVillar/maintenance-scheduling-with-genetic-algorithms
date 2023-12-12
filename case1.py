import numpy as np

# Define the problem parameters
num_units = 7
num_intervals = 4
max_loads = [80, 90, 65, 70]
totalcap = 150  # Initialize totalcap

# Define the GA parameters
pop_size = 10
num_generations = 1000
mutation_rate = 0.01

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

# Generate a chromosome
# chromosome = generateGenome(unitData, num_intervals)

# Calculate the fitness of the chromosome
def calculate_fitness(chromosome, unitData, num_intervals, max_loads, totalcap):
    net_reserves = np.zeros(num_intervals, dtype=int)
    numChromosome = len(chromosome)

    for col in range(num_intervals):
        totalLoad = 0

        # totalLoad is the sum of capacities of the units scheduled for maintenance at each interval
        for row in range(numChromosome):
            totalLoad += (chromosome[row, col] * unitData[row][0])

        net_reserves[col] = totalcap - totalLoad - max_loads[col]

    # if the net reserve at any interval is negative, the schedule is illegal, and the fitness function returns zero
    if any(net_reserves < 0):
        return 0
    
    # define the fitness value as the lowest net reserve    
    return np.min(net_reserves)

# Generate Fitness of the chromosome
# fitness = calculate_fitness(chromosome, unitData, num_intervals, max_loads, totalcap)

def initialize_population(pop_size, unitData, num_intervals, max_loads, totalcap):
    population = []
    for _ in range(pop_size):
        chromosome = generateGenome(unitData, num_intervals)
        fitness_value = calculate_fitness(chromosome, unitData, num_intervals, max_loads, totalcap)
        population.append({"chromosome": chromosome, "fitness": fitness_value})
    return population

# Generate the population
population = initialize_population(pop_size, unitData, num_intervals, max_loads, totalcap)

for i, individual in enumerate(population):
    print(f"Individual {i + 1}:")
    print("Chromosome:")
    print(individual["chromosome"])
    print("Fitness:", individual["fitness"])
    print()
