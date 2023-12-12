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

def spCrossover(parent1, parent2, crossoverProbability):
  numUnits = len(unitData)
  crossoverPoint = np.random.randint(1, numUnits)
  
  child1 = parent1[:crossoverPoint] + parent2[crossoverPoint:]
  child2 = parent2[:crossoverPoint] + parent1[crossoverPoint:]

  return child1, child2

populationSize = 20
mutationRate = 0.01
crossoverProbability = 0.7
generations = 10

# initial population
# Population = np.array([generateGenome(unitData, intervals) for _ in range(populationSize)])

# fitnessValues = np.zeros((generations, populationSize), dtype=int)

# for generation in range(generations):

#   # calc fitness for each chromosome in population
#   for i, chromosome in enumerate(Population):
#     fitnessValues[generation, i] = fitness(chromosome,totalCap,unitData,intervals,intervalLoad)
#   print(fitnessValues[generation,i])


# totalFitness = np.sum(fitnessValues)
# generate choromosome
# chromosome = generateGenome(unitData, intervals)

#print(f"generate chromose: ")
#print(chromosome)
# print(totalCap)
# fitness(chromosome, totalCap, unitData, intervals, intervalLoad)
# wanako kasabot