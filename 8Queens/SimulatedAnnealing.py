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
best_state, best_fitness = mlrose.simulated_annealing(problem_fit, random_state = 10, max_attempts = 20)
#best_state, best_fitness = mlrose.simulated_annealing(problem_fit, max_attempts = 10)

stop = time.time()
print(best_state)

print(best_fitness)
print(stop-start)
