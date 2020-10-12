import mlrose
import numpy as np
import time

# Create list of city coordinates
init_state = np.array([0,1,2,3,4,5,6,7])

# Initialize fitness function object using coords_list
fitness = mlrose.Queens()

# Define optimization problem object
problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)

start = time.time()
# Solve problem using the genetic algorithm
best_state, best_fitness = mlrose.random_hill_climb(problem_fit, random_state = 15, max_attempts = 250)
#best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_attempts = 10)

stop = time.time()
print(best_state)

print(best_fitness)
print(stop-start)

#max_attempts anything after 11 until 1000 gives the same best_fitness value with length of 8
#we are minimizing the fitness function, which means less distance(units) tranvelled
#increasing the length of coordinates from 8,10 with the same parameters values
