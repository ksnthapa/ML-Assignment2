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
#best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 5, max_attempts = 200, mutation_prob = 0.2)
best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state=20, max_attempts = 100, mutation_prob = 0.4)
stop = time.time()

print(best_state)
print(best_fitness)
print(stop-start)

#max_attempts anything after 11 until 1000 gives the same best_fitness value with length of 8
#we are minimizing the fitness function, which means less distance(units) tranvelled
#increasing the length of coordinates from 8,10 with the same parameters values
