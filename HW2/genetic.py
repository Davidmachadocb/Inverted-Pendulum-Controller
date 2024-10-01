import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np
import random

def fitness(controller_constants, derivatives, t, s0):
    solution = integrate.odeint(func=derivatives, y0=s0, t=t, args=(controller_constants,))

    theta_error = np.sum(solution[:, 0]**2)
    x_error = np.sum(solution[:, 2]**2)
    return theta_error + x_error

def genetic_algorithm(pop_size, num_generations, mutation_rate, derivatives, s0, t):
    population = [np.random.uniform(0, 55, 4) for _ in range(pop_size)]

    for generation in range(num_generations):
        # Evaluate fitness for the population
        fitness_scores = [fitness(individual, derivatives, t, s0) for individual in population]

        # Select the best individuals to be parents
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
        population = sorted_population[:pop_size // 2]

        # Create next generation through crossover
        next_generation = []
        while len(next_generation) < pop_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            next_generation.append(child)

        # Apply mutation
        for individual in next_generation:
            if random.random() < mutation_rate:
                mutation_index = random.randint(0, len(individual) - 1)
                individual[mutation_index] += random.uniform(-1, 1)

        population = next_generation

    # Evaluate fitness for the final population
    fitness_scores = [fitness(individual, derivatives, t, s0) for individual in population]

    # Return the best individual
    best_individual = population[np.argmin(fitness_scores)]
    return best_individual
