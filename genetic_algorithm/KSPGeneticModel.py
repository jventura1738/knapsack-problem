# Justin Ventura

# Import numerical libraries:
import numpy as np


# This is the GeneticModel class:
class KSPGeneticModel:
    '''
    This KSP Genetic Model uses:
        1) Tournament Selection
        2) 1/2 Crossover Point
        3) 2.5% Chance for Bit Flip Mutation
    '''

    # Constructor for the GeneticModel class:
    def __init__(self, initial_population, threshold, weight_vector, value_vector):
        '''
        Context: The intial population is a list of potential solutions to the
        Knap Sack Problem. The threshold is the maximum weight that the knapsack.
        The weight vector is the weight of each item. The value vector is the value
        of each item.

        Throws assertion errors for invalid KSP parameters.
        '''
        # Validate KSP here:
        assert initial_population is not None, 'Initial population cannot be None!'
        assert threshold and threshold > 0, 'Threshold must be greater than 0!'
        assert len(weight_vector) == len(
            value_vector), 'Weight and value vectors must be the same length!'

        # Intialize the main parameters:
        self.population = initial_population
        self.fitness = np.zeros(len(initial_population))
        self.threshold = threshold
        self.weights = weight_vector
        self.values = value_vector

        # These are essentially constants:
        self.crossover_point = int(len(weight_vector) / 2)
        self.generations = 0

        # Calculate the intial fitness of the population:
        self.calculate_fitness()

        # This keeps track of current performance:
        self.best_chromosome, self.best_chromosome_fitness = self.get_best_chromosome()
        self.average_fitness = self.get_average_fitness()

        # This keeps track of the best ever performance:
        self.best_ever_chromosome = self.best_chromosome
        self.best_ever_chromosome_fitness = self.best_chromosome_fitness
        self.best_average_fitness = self.average_fitness
        self.best_chromosome_generation = 0
        self.best_average_fitness_generation = 0

    # Return the summary of the generation:
    def get_current_summary_statistics(self) -> dict:
        '''Return a dictionary of summary statistics'''
        return {
            'best_chromosome': self.best_chromosome,
            'best_chromosome_fitness': self.best_chromosome_fitness,
            'average_fitness': self.average_fitness,
            'current_generation': self.generations,
        }

    # Get the historically best values from the model:
    def get_historical_summary(self) -> dict:
        '''Return the best performance of the algorithm'''
        return {
            # These are the historical values:
            'best_ever_chromosome': self.best_ever_chromosome,
            'best_ever_chromosome_fitness': self.best_ever_chromosome_fitness,
            'best_chromosome_generation': self.best_chromosome_generation,
            'best_average_fitness': self.best_average_fitness,
            'best_average_fitness_generation': self.best_average_fitness_generation,
        }

    # Update the summary statistics:
    def update_summary_statistics(self) -> None:
        '''Update the summary statistics of the genetic algorithm'''
        # Update the best chromosome:
        self.best_chromosome, self.best_chromosome_fitness = self.get_best_chromosome()
        if self.best_chromosome_fitness > self.best_ever_chromosome_fitness:
            self.best_ever_chromosome = self.best_chromosome
            self.best_ever_chromosome_fitness = self.best_chromosome_fitness
            self.best_chromosome_generation = self.generations

        # Update the best average fitness:
        self.average_fitness = self.get_average_fitness()
        if self.average_fitness > self.best_average_fitness:
            self.best_average_fitness = self.average_fitness
            self.best_average_fitness_generation = self.generations

    # Get the current best chromosome:
    def get_best_chromosome(self) -> list:
        '''Return the best chromosome (first occurrence)'''
        best, best_idx = self.population[0], 0
        for i in range(1, len(self.population)):
            if self.fitness[i] > self.fitness[best_idx]:
                best, best_idx = self.population[i], i
        return best, self.fitness[best_idx]

    # Return the average fitness of the population:
    def get_average_fitness(self) -> float:
        '''Return the average fitness of the population'''
        return np.mean(self.fitness)

    # Run a generation:
    def run_generation(self, debug: bool = False) -> list:
        '''Run a generation of the genetic algorithm'''

        # This is for logging purposes:
        if debug:
            print('After calculate_fitness()\n', self, end='\n\n', sep='')
            print('Tournament beginning...')

        # PHASE 1: Select the chromosomes for the next generation:
        self.population = self.select_chromosomes(debug=debug)

        if debug:
            self.calculate_fitness()
            print('\nAfter select_chromosomes()\n', self, end='\n\n', sep='')

        # PHASE 2: Cross over the chromosomes:
        self.population = self.crossover_chromosomes()

        if debug:
            self.calculate_fitness()
            print('After crossover_chromosomes()\n', self, end='\n\n', sep='')

        # PHASE 3: Mutate the chromosomes:
        self.mutate_chromosomes(debug=debug)

        if debug:
            self.calculate_fitness()
            print('\nAfter mutate_chromosomes()\n', self, end='\n\n', sep='')

        # Calculate the intial fitness of the population:
        self.calculate_fitness()

        # Increment the generation count and return the new population:
        self.generations += 1
        self.update_summary_statistics()
        return self.population

    # Calculate the fitness of each chromosome:
    def calculate_fitness(self) -> None:
        '''Fitness function based on the Knap Sack Problem Definition'''
        for i in range(len(self.population)):
            self.fitness[i] = self.calculate_chromosome_fitness(
                self.population[i])

    # Calculate the fitness of a chromosome:
    def calculate_chromosome_fitness(self, chromosome):
        '''A simple dot product of the item vectors and the chromosome'''
        if np.dot(chromosome, self.weights) > self.threshold:
            return 0
        else:
            return np.dot(chromosome, self.values)

    # Select the chromosomes for the next generation with tournament selection:
    def select_chromosomes(self, debug: bool = False) -> list:
        '''This selection uses a 1v1 tournament selection method'''
        new_population = list()
        for _ in range(len(self.population)):

            # Select two random chromosomes:
            idx1 = np.random.randint(0, len(self.population))
            idx2 = np.random.randint(0, len(self.population))
            chromosome_1 = self.population[idx1]
            chromosome_2 = self.population[idx2]

            if debug:
                print(f'Chromosome 1: {chromosome_1} -> {self.fitness[idx1]}')
                print(f'Chromosome 2: {chromosome_2} -> {self.fitness[idx2]}')

            # Select the chromosome with the higher fitness:
            if self.fitness[idx1] > self.fitness[idx2]:
                new_population.append(chromosome_1)
            else:
                new_population.append(chromosome_2)

        return new_population

    # Cross over the chromosomes:
    def crossover_chromosomes(self) -> list:
        '''This crossover uses a 1/2 crossover point'''
        new_population = list()
        # Iterate through the parents selected:
        for i in range(0, len(self.population) - 1, 2):
            # Catch edge case:
            if i + 1 > len(self.population):
                break
            parent1 = self.population[i]
            parent2 = self.population[i + 1]

            child1 = np.concatenate(
                (parent1[:self.crossover_point], parent2[self.crossover_point:]))
            child2 = np.concatenate(
                (parent2[:self.crossover_point], parent1[self.crossover_point:]))

            new_population.append(child1)
            new_population.append(child2)

        return new_population

    # Mutate the chromosomes:
    def mutate_chromosomes(self, debug: bool = False) -> None:
        '''This mutation uses a 2.5% chance for a bit flip mutation'''
        # Iterate through the chromosomes and flip a bit with a 5% chance:
        for i, chromosome in enumerate(self.population):
            for j in range(len(chromosome)):
                if np.random.rand() < 0.025:
                    # Flip the bit:
                    chromosome[j] = 1 - chromosome[j]
                    if debug:
                        print('A mutation occured on chromosome',
                              i, 'at index', j, '!')

    # Return the number of generations:
    def get_generations(self) -> int:
        '''Return the number of generations'''
        return self.generations

    # String representation of the object:
    def __str__(self) -> str:
        result_string = list()
        for c, f in zip(self.population, self.fitness):
            result_string.append(f'Chromosome: {c} Fitness: {f}')
        return '\n'.join(result_string)

    # Return the object representation:
    def __repr__(self) -> str:
        return f'KSPGeneticModel(population={self.population}, fitness={self.fitness}, threshold={self.threshold}, weights={self.weights}, values={self.values})'
