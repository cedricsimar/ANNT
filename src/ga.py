
from settings import Settings

from dna import DNA
from mutation import Mutation
from cross_over import Cross_Over
from nn import NN

class GeneticAlgorithm(object):

    def __init__(self, population_size, mutations_per_generation):
        
        self.generation = 0
        self.population_size = population_size
        self.mutations_per_generation = mutations_per_generation

        self.population = []
        self.neural_networks = []
        self.fitness = []
        
        self.best_fitness_score = 0
        
        # individuals that are "saved" and go through the next generation unchanged
        self.best_individuals = []  

        self.create_initial_population()


    def evolve(self, num_generations):
        
        """
        Evolve the population for a fixed number of generations
        """

        for gen in range(num_generations):

            self.generation += 1
            print("Evolving generation " + str(self.generation))

            # save the 3 best individuals

            # for each individual
            #   - cross-over and mutate
            #   - repair if necessary
            #   - create the corresponding neural network,
            #   - train it on the MNIST dataset and store it's fitness score
            
            for individual_i in range(len(self.population)):

                # cross-over
                # offspring_1, offspring_2 = Cross_Over()
                
                # mutate
                mutant = Mutation(self.population[individual_i]).mutate()

                # repair

                # create the corresponding neural network
                print(mutant)
                nn = NN(mutant)

                # compute fitness score and update population

            # keep the best individuals in the next generation


    def create_initial_population(self):

        """
        Create an initial population consisting of *population_size*
        primitive neural networks with an input-output structure
        """

        for individual in range(self.population_size):
            self.population.append(DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE))