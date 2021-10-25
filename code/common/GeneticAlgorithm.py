import numpy as np
import time

from numpy.lib.function_base import select 

def default_individual_constraint(pop):
    return 
class GeneticAlgorithm:
    def __init__(self, param_size: int, pop_size: int, low_bound: int, high_bound: int,
                fitness_func, constraint_func, discrete=False):
        self.param_size = param_size
        self.pop_size = pop_size
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.initial_high_bound = high_bound
        self.fitness_func = fitness_func # input is total_pop
        self.environmental_constraint = constraint_func
        self.discrete = discrete
        self.n_mating = pop_size//2
        self.crossover_idx = param_size//2
        self.mutation_rate = 0.5
        self.mutation_center = 0
        self.mutation_delta = 0.5
        
        self.best_solution = np.array([])
        self.best_outputs = []
        self.runtime = -1

    def create_random_pop(self, num_indivuals):
        population = np.random.uniform(
            low = self.low_bound, 
            high = self.initial_high_bound, 
            size=(num_indivuals, self.param_size))
        if self.discrete:
            mu = 0.0 # mean
            sigma = 0.3 # st. dev, spread
            # population = np.random.normal(loc = mu, scale=sigma, size = (num_indivuals, self.param_size))
            
            population = np.round(population)
        assert population.shape == (num_indivuals, self.param_size)
        return population

    def gen_population(self):
        """
        Generates a random matrix of size (pop_size, param_size) with values in the given bounds. 
        Integer values are used if discrete = True
        """
        population = self.create_random_pop(self.pop_size)
        population = self.environmental_constraint(population)
        while population.shape[0] < self.pop_size:
            print(f"Cur Shape: {population.shape[0]}")
            cur_size = population.shape[0]
            extra_pop = self.create_random_pop(self.pop_size-cur_size)
            population = np.vstack((population, extra_pop))
            population = self.environmental_constraint(population)

        assert population.shape == (self.pop_size, self.param_size)
        return population

    def calculate_fitness(self, total_pop):
        fitness = self.fitness_func(total_pop)
        assert fitness.shape == (total_pop.shape[0],)
        return fitness
    
    def select_deterministic(self, total_pop, select_num):
        parents = np.empty((select_num, self.param_size))
        t_fitness = self.calculate_fitness(total_pop)
        best_fitness = np.argsort(t_fitness)[0:select_num]
        parents = total_pop[best_fitness]
        assert parents.shape == (select_num, self.param_size)
        return parents

    def crossover(self, t_parents):
        n_offspring = self.n_mating - 1
        # Create an emppty vector
        offspring = np.empty((n_offspring, t_parents.shape[1]))

        # Fill in crossover details
        for i in range(n_offspring):
            parent1 = t_parents[i]
            parent2 = t_parents[i+1]
            offspring[i] = np.copy(parent2)
            offspring[i,self.crossover_idx:] = parent1[self.crossover_idx:]
        
        assert offspring.shape == (n_offspring, self.param_size)
        return offspring
    
    def mutation(self, t_offspring):
        mutated_offspring = np.copy(t_offspring)
        random_values = np.random.uniform(
            low=self.mutation_center-self.mutation_delta, 
            high=self.mutation_center+self.mutation_delta, 
            size=t_offspring.shape
        )
        coin_toss = np.random.uniform(low=0, high=1, size=t_offspring.shape)
        ind = np.nonzero(coin_toss > self.mutation_rate)
        if self.discrete:
            mutated_offspring[ind] += np.round(random_values[ind])
        else:
            mutated_offspring[ind] += random_values[ind]
        # Clip mutations between low and high bounds
        too_low_idx = mutated_offspring < self.low_bound
        too_high_idx = mutated_offspring > self.high_bound
        mutated_offspring[too_low_idx] = self.low_bound
        mutated_offspring[too_high_idx] = self.high_bound

        assert ((mutated_offspring >= self.low_bound) & (mutated_offspring <= self.high_bound)).all()
        assert mutated_offspring.shape == t_offspring.shape
        return mutated_offspring
    
    def environmental_selection(self, curr_population, offspring):
        offspring = self.environmental_constraint(offspring)
        print(f"Offspring: {offspring}")
        t_total_pop = np.vstack((curr_population, offspring))
        new_pop = self.select_deterministic(t_total_pop, self.pop_size)
        assert new_pop.shape == (self.pop_size, self.param_size)
        return new_pop

    def run(self, num_generations):
        curr_population = self.gen_population()
        self.best_outputs = []
        self.avg_fitness = []
        overall_min_fitness = 999999
        start_time = time.time()
        for generation in range(num_generations):
            print(f"Generation : {generation}")
            # print(f"Current Population: {curr_population}")
            fitness = self.calculate_fitness(curr_population)
            min_fitness = np.min(fitness)

            overall_min_fitness = min(min_fitness, overall_min_fitness)
            print(f"Best result in current iteration {min_fitness} compared to overall {overall_min_fitness}")
            self.best_outputs.append(min_fitness)
            self.avg_fitness.append(np.average(fitness))
            # Selecting the best parents in the population for mating.
            parents = self.select_deterministic(curr_population, self.n_mating)
            offspring = self.crossover(parents)
            offspring = self.mutation(offspring)
            # Environmental selection
            curr_population = self.environmental_selection(curr_population, offspring)
        fitness = self.calculate_fitness(curr_population)
        print(fitness)
        # Then return the index of that solution corresponding to the best fitness.
        min_idx = np.argmin(fitness)
        self.best_solution = curr_population[min_idx]
        print("Best solution : ", curr_population[min_idx])
        print("Best solution fitness : ", fitness[min_idx])
        end_time = time.time()
        self.runtime = end_time-start_time
