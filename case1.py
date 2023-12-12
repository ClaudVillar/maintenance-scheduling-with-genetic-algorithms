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


def generateGenome(unitData, intervals, totalCap):
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

    totalCap = totalCap + unitData[unit][0]

  return chromosome, totalCap

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


#generate choromosome
chromosome, totalCap = generateGenome(unitData, intervals, totalCap)

print(f"generate chromose: ")
print(chromosome)
print(totalCap)
fitness(chromosome, totalCap, unitData, intervals, intervalLoad)
# print(fitness(chromosome,totalCap,unitData,intervals,intervalLoad))
#final test