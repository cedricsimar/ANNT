
from settings import Settings

from dna import DNA
from nn import NN

class GeneticAlgorithm(object):

    def __init__(self, population_size, mutation_rate):

        self.population_size = population_size
        self.mutation_rate = mutation_rate

        self.population = []
        self.neural_networks = []
        self.fitness = []
        
        self.create_initial_population()


    def evolve(self, num_generations):
        
        """
        Evolve the population for a fixed number of generations
        """
        print("Evolving")

        # for each individual, create the corresponding neural network,
        # train it on the MNIST dataset and store it's fitness score
        
        for individual_dna in self.population:

            # create the corresponding neural network
            print(individual_dna)
            nn = NN(individual_dna)


    def create_initial_population(self):

        """
        Create an initial population consisting of *population_size*
        primitive neural networks with an input-output structure
        """

        for individual in range(self.population_size):
            self.population.append(DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE))