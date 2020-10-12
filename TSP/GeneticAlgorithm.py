import time

import mlrose
import numpy as np

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (5, 9), (2, 6)]
#coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (5, 9), (2, 6), (7, 5), (8, 6), (7, 1), (5, 7), (6, 2), (6, 8)]

# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)

# Define optimization problem object
problem_fit = mlrose.TSPOpt(length = 10, fitness_fn = fitness_coords, maximize=False)

start = time.time()

# Solve problem using the genetic algorithm
#best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 10, max_attempts = 15, mutation_prob = 0.2)
best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 15, max_attempts = 20, mutation_prob = 0.2)


stop = time.time()

print(best_state)

print(best_fitness)
print(stop-start)
#max_attempts anything after 11 until 1000 gives the same best_fitness value with length of 8
#we are minimizing the fitness function, which means less distance tranvelled
#increasing the length of coordinates from 8,10 with the same parameters values
