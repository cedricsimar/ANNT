
from settings import Settings

from dna import DNA
from mutation import Mutation
from cross_over import Cross_Over
from nn import NN

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GeneticAlgorithm(object):

    def __init__(self, population_size, mutations_per_generation):
        
        self.generation = 0
        self.population_size = population_size
        self.mutations_per_generation = mutations_per_generation

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.validation_data = self.mnist.test.images
        self.validation_labels = self.mnist.test.labels

        self.population = []
        self.neural_networks = []
        self.fitness = []
        
        self.best_fitness_score = 0

        self.create_initial_population()


    def evolve(self, num_generations):
        
        """
        Evolve the population for a fixed number of generations
        """

        for gen in range(num_generations):

            self.generation += 1
            print("Evolving generation " + str(self.generation))

            self.next_generation_dna = []
            self.next_generation_nn = []
            self.next_generation_fitness = []

            # save the best individuals that go through the next generation unchanged

            # for each individual
            #   - cross-over and mutate
            #   - repair if necessary
            #   - create the corresponding neural network,
            #   - train it on the MNIST dataset and store it's best fitness score
            #   - apply natural selection in the population
            
            for individual_i in range(len(self.population)):

                # cross-over
                # offspring_1, offspring_2 = Cross_Over()
                
                # mutation
                mutant = Mutation(self.population[individual_i]).mutate()
                self.next_generation_dna.append(mutant)
                print(mutant)

                # create the corresponding neural network
                nn = NN(mutant)
                self.next_generation_nn.append(nn)

                # train the neural network
                best_validation_error = 1
                sess = tf.Session(graph=nn.get_graph())
                sess.run(tf.global_variables_initializer())

                for training_step in range(Settings.TRAINING_STEPS):
                    
                    # training step
                    images, labels = self.mnist.train.next_batch(Settings.MINIBATCH_SIZE)
                    sess.run(nn.optimize, {nn.input: images, nn.labels: labels})

                    # performance evaluation every x steps
                    if not training_step % Settings.EVALUATION_RATE:
                        validation_error = sess.run(nn.prediction_error, {nn.input: self.validation_data, nn.labels = self.validation_labels})
                        print('Test error {:6.2f}%'.format(100 * validation_error))
                        best_validation_error = min(best_validation_error, validation_error)


                # compute fitness score and update population

                # save graph logs
                writer = tf.summary.FileWriter("./tmp/log", sess.graph)
                writer.close()

            # keep the best individuals in the next generation


    def create_initial_population(self):

        """
        Create an initial population consisting of *population_size*
        primitive neural networks with an input-output structure
        """

        for individual in range(self.population_size):
            
            dna = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)
            dna.create_primitive_structure()
            self.population.append(dna)