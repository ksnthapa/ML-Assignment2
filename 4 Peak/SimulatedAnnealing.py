import mlrose
import numpy as np
import time

# Create list of city coordinates
state = np.array([1,1,1,0,1,0,0,1,0,0,0,0,1,1,1,1])

fitness = mlrose.FourPeaks(t_pct=0.2)

# Define optimization problem object
problem_fit = mlrose.DiscreteOpt(length = 16, fitness_fn = fitness, maximize=True)

start = time.time()
# Solve problem using the genetic algorithm
best_state, best_fitness = mlrose.simulated_annealing(problem_fit, random_state = 10, max_attempts = 20)
#best_state, best_fitness = mlrose.simulated_annealing(problem_fit, max_attempts = 10)

stop = time.time()
print(best_state)

print(best_fitness)
print(stop-start)
