# Justin Ventura

# Import numerical libraries:
import numpy as np

# Import visualization libraries:
import matplotlib.pyplot as plt

# Import the KSP Genetic Model:
from genetic_algorithm.KSPGeneticModel import KSPGeneticModel


# ----------------------------------------------------------
# THIS IS THE INFORMATION FOR THE KNAPSACK PROBLEM:
weight_vector = [1, 2, 3, 2, 1, 3, 5, 1, 4, 2, 8, 5, 3, 6, 2]
value_vector = [4, 4, 5, 1, 1, 6, 8, 3, 2, 1, 10, 2, 1, 1, 9]
NUM_ITEMS = len(weight_vector)
POPULATION_SIZE = 10
THRESHOLD = 20
initial_population = [np.random.randint(
    0, 2, size=NUM_ITEMS) for _ in range(POPULATION_SIZE)]
TheKnapSackProblem = {
    'initial_population': initial_population,
    'threshold': THRESHOLD,
    'weight_vector': weight_vector,
    'value_vector': value_vector,
}
# ----------------------------------------------------------

# Controlling the simulation:
NUM_GENERATIONS = 300
DEBUG = False

# Main function that demonstrates the algorithm:


def _main():
    SimulationModel = KSPGeneticModel(**TheKnapSackProblem)
    best_chromosomes, average_fitnesses, generations = list(), list(), list()

    # Run NUM_GENERATIONS generations:
    for _ in range(NUM_GENERATIONS):
        SimulationModel.run_generation(debug=DEBUG)
        _, best_chromosome, average_fitness, generation = SimulationModel.get_current_summary_statistics().values()
        best_chromosomes.append(best_chromosome)
        average_fitnesses.append(average_fitness)
        generations.append(generation)

    # Print the hisorical summary:
    historical_summary = SimulationModel.get_historical_summary()
    print(historical_summary)

    # Plot the results of the model:
    plt.title('Optimized Knapsack Value vs. Generations')
    plt.plot(generations, best_chromosomes, label='Best Chromosome')
    plt.plot(generations, average_fitnesses, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Knapsack Value')
    plt.legend()
    plt.show()


# Run the main function if this file is run as a script.
if __name__ == '__main__':
    _main()
