import mlrose
import numpy as np
import time

# Create list of city coordinates
state = np.array([1,1,1,0,1,0,0,1,0,0,0,0,1,1,1,1])
fitness = mlrose.FourPeaks(t_pct=0.15)

# Define optimization problem object
problem_fit = mlrose.DiscreteOpt(length = 16, fitness_fn = fitness, maximize=True, max_val=2)
start = time.time()
# Solve problem using the genetic algorithm
best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 15, max_attempts = 50, mutation_prob = 0.2)

stop = time.time()
print(best_state)

print(best_fitness)
print(stop-start)

#max_attempts anything after 11 until 1000 gives the same best_fitness value with length of 8
#we are minimizing the fitness function, which means less distance(units) tranvelled
#increasing the length of coordinates from 8,10 with the same parameters values
