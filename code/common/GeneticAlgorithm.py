import numpy as np
import time 

class GeneticAlgorithm:
    def __init__(self, param_size: int, pop_size: int, low_bound: int, high_bound: int,
                fitness_func, discrete=False):
        self.param_size = param_size
        self.pop_size = pop_size
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.fitness_func = fitness_func # input is total_pop

        self.discrete = discrete
        self.n_mating = pop_size//2
        self.crossover_idx = param_size//2
        self.mutation_rate = 0.5
        self.mutation_delta = 0.5

        self.best_outputs = []
        self.runtime = -1

    def gen_population(self):
        """
        Generates a random matrix of size (pop_size, param_size) with values in the given bounds. 
        Integer values are used if discrete = True
        """
        population = np.random.uniform(
            low = self.low_bound, 
            high = self.high_bound, 
            size=(self.pop_size, self.param_size))
        if self.discrete:
            population = np.round(population)
        return population

    def calculate_fitness(self, total_pop):
        return self.fitness_func(total_pop)
    
    def select_deterministic(self, total_pop, select_num):
        parents = np.empty((select_num, self.param_size))
        t_fitness = self.calculate_fitness(total_pop)
        best_fitness = np.argsort(t_fitness)[0:select_num]
        parents = total_pop[best_fitness]
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
            low=-self.mutation_delta, 
            high=self.mutation_delta, 
            size=t_offspring.shape
        )
        coin_toss = np.random.uniform(low=0, high=1, size=t_offspring.shape)
        ind = np.nonzero(coin_toss > self.mutation_rate)
        print(ind)
        print(random_values.shape)
        mutated_offspring[ind] += random_values[ind]

        assert mutated_offspring.shape == t_offspring.shape
        return mutated_offspring
    
    def environmental_selection(self, curr_population, offspring):
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
            print(f"Current Population: {curr_population}")
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
        max_idx = np.argmin(fitness)

        print("Best solution : ", curr_population[max_idx])
        print("Best solution fitness : ", fitness[max_idx])
        end_time = time.time()
        self.runtime = end_time-start_time
