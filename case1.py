import numpy as np
totalCap = 0
intervals = 4
intervalLoad = [80,90,65,70]
unitData = np.array([
    (20, 2),
    (15, 2),
    (35, 1),
    (40, 1),
    (15, 1),
    (15, 1),
    (10, 1)
])


populationSize = 20
mutationRate = 0.001
crossoverRate = 0.7
numGenerations = 50


totalCap = sum(unitData[i][0] for i in range(len(unitData)))

def generateGenome(unitData, intervals):
  # empty 
  numUnits = len(unitData)
  chromosome = np.zeros((numUnits, intervals), dtype=int)
  
  for unit in range(numUnits):
    if unitData[unit][1] > intervals:
      raise ValueError("Unit's required interval must not be greater than load interval")
    
    # maintenance of any unit starts at the beginning of an interval and finishes at the same or adjacent interval. example required interval for unit = 2 then [0 0 1 1] [1 1 0 0 ] [0 1 1 0]    
    startInterval = np.random.randint(0, intervals - unitData[unit][1] + 1)

    for interval in range(startInterval, startInterval + unitData[unit][1]):
      chromosome[unit, interval] = 1

  return chromosome


def fitness(chromosome, totalCap, unitData, intervals, intervalLoad):
  netReserve = np.zeros(intervals, dtype=int)
  numChromosome = len(chromosome)

  for col in range(intervals):
    # totalLoad is the sum of capacities of the units scheduled for maintenance at each interval
    totalLoad = 0

    for row in range(numChromosome):
      totalLoad += (chromosome[row, col] * unitData[row][0])
    
    netReserve[col] = totalCap - totalLoad - intervalLoad[col]
  
  # if the net reserve at any interval is negative, the schedule is illegal, and the fitness function returns zero
  if any(netReserve < 0):
    return 0
  
  # the fitness value is the lowest net reserve
  return np.min(netReserve)

def rouletteSelection(Population, fitnessValues):
  fitnessRatio = fitnessValues / np.sum(fitnessValues)
  selectedIndex = np.random.choice(len(Population), p=fitnessRatio)

  return Population[selectedIndex]["chromosome"]

def spCrossover(parent1, parent2, crossoverRate, intervals):
  if np.random.rand() < crossoverRate:
    numUnits, intervals = parent1.shape
    crossoverPoint = np.random.randint(1, numUnits)
  
    child1 = np.concatenate((parent1[:crossoverPoint], parent2[crossoverPoint:]))
    child2 = np.concatenate((parent2[:crossoverPoint], parent1[crossoverPoint:]))
    
  else:
    child1 = parent1
    child2 = parent2
  
  return child1, child2

def mutation(chromosome, mutationRate):
    mutated = np.copy(chromosome)
    mutationMask = (np.random.rand(*chromosome.shape) < mutationRate).astype(int)

    for unit in range(len(unitData)):
        requiredMaintenance = unitData[unit][1]

        # Ensure that the mutation maintains the required maintenance intervals
        for i in range(intervals - requiredMaintenance + 1):
            if mutationMask[unit, i:i + requiredMaintenance].all():
                mutated[unit, i:i + requiredMaintenance] = 1

    return mutated

# initial population
def initPopulation(populationSize, unitData, intervals, intervalLoad, totalCap):
  Population = []
  fitnessValues = np.zeros(populationSize, dtype=int)
  for i in range(populationSize):
    chromosome = generateGenome(unitData, intervals)
    Population.append({"chromosome": chromosome, "fitness": fitness(chromosome, totalCap, unitData, intervals, intervalLoad)})
    fitnessValues[i] = Population[i]["fitness"]

  return Population, fitnessValues
  # print(fitnessValues)
  # fitnessTotal = np.sum(fitnessValues)
  # print(fitnessTotal)

def geneticAlgo(unitData, intervals, intervalLoad, totalCap, populationSize, numGenerations, mutationRate, crossoverRate):
  # initialize Population
  Population, fitnessValues = initPopulation(populationSize, unitData, intervals, intervalLoad, totalCap)

  for generation in range(numGenerations):
    
    parents = [rouletteSelection(Population, fitnessValues) for _ in range(2)]
    
    # crossover process: uses single-point crossover 
    child1, child2 = spCrossover(parents[0], parents[1], crossoverRate, intervals)

    # mutation process
    child1 = mutation(child1, mutationRate)
    child2 = mutation(child2, mutationRate)

    # calculates fitness for the children
    fitnessChild1 = fitness(child1, totalCap, unitData, intervals, intervalLoad)
    fitnessChild2 = fitness(child2, totalCap, unitData, intervals, intervalLoad)

    # replaces the weak individuals in the population with a fitter child
    weakIndex = np.argmin(fitnessValues)
    Population[weakIndex] = {"chromosome": child1, "fitness": fitnessChild1}
    Population[(weakIndex + 1) % populationSize] = {"chromosome": child2, "fitness": fitnessChild2}

    # update fitness values to be used with the fitness values of the newly generated children
    fitnessValues[weakIndex] = fitnessChild1  
    fitnessValues[(weakIndex + 1) % populationSize] = fitnessChild2

    # prints the best individual in the current generation
    best_individual = max(Population, key=lambda x: x["fitness"])
    print(f"Generation {generation + 1}, Best Fitness: {best_individual['fitness']}")

  return Population

finalPopulation = geneticAlgo(unitData, intervals, intervalLoad, totalCap, populationSize, numGenerations, mutationRate, crossoverRate)

# Print the best individual in the final population
best_individual = max(finalPopulation, key=lambda x: x["fitness"])
print("\nBest Individual in the Final Population:")
print("Chromosome:")
print(best_individual["chromosome"])
print("Fitness:", best_individual["fitness"])