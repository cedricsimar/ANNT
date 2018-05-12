
from settings import Settings
from exceptions import NoBridgeException
from random import randint

from dna import DNA
from mutation import Mutation
from cross_over import Cross_Over
from nn import NN
from utils import reshape_mnist_images

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GeneticAlgorithm(object):

    def __init__(self, population_size, mutations_per_generation):
        
        self.generation = 0
        self.population_size = population_size
        self.mutations_per_generation = mutations_per_generation

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        
        self.validation_data = reshape_mnist_images(self.mnist.test.images)
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

            # save the three best individuals that go through the next generation unchanged
            # (produces 3 offsprings)
            for best_i in range(Settings.N_BEST_CANDIDATES):

                self.next_generation_dna.append(self.population[best_i])
                self.next_generation_nn.append(self.neural_networks[best_i])
                self.next_generation_fitness.append(self.fitness[best_i])

            # breed the three best individuals together
            # (produces at most 6 offsprings)
            for best_i in range(Settings.N_BEST_CANDIDATES):
                for best_j in range(best_i + 1, Settings.N_BEST_CANDIDATES):
                    self.attempt_breeding(best_i, best_j)
            
            # breed each best individual with a random individual from the population
            # (produces at most 6 offsprings)
            for best_i in range(Settings.N_BEST_CANDIDATES):
                other_i = randint(0, self.population_size - 1)
                self.attempt_breeding(best_i, other_i)

            # breed random individuals together until the number of offsprings reaches 
            # the maximum population size
            while(len(self.next_generation_dna) < self.population_size):
                other_i = randint(0, self.population_size - 1)
                other_j = randint(0, self.population_size - 1)
                self.attempt_breeding(other_i, other_j)


            # train every non-trained neural network of the next generation
            # on the MNIST dataset and store it's best fitness score

            # for individual_i in range(Settings.N_BEST_CANDIDATES, self.population_size):

            #     # train the neural network and store the individual's fitness score
            #     individual_fitness = self.train_network(self.next_generation_nn[individual_i])
            #     self.next_generation_fitness[individual_i] = individual_fitness
            
            # sort the next generation individuals by fitness
            self.next_generation_fitness, self.next_generation_dna, self.next_generation_nn = (list(t) for t in zip(*sorted(zip(self.next_generation_fitness, self.next_generation_dna, self.next_generation_nn))))

            # update the population for the next generation
            self.population = self.next_generation_dna
            self.neural_networks = self.next_generation_nn
            self.fitness = self.next_generation_fitness


    def attempt_breeding(self, individual_i, individual_j):

        """
        Attempt to breed the two individuals (several times if necessary)
        If the breeding is successful and the Neural Networks can be built,
        the two offsprings are added into the next generation.

        The breeding process is the following:
            - cross-over between the two lovers
            - mutate the offsprings with probability p
            - create the offsprings neural networks

        """

        successful_breedings = 0
        breeding_attempts = 0
        while(successful_breedings < 2 and breeding_attempts < Settings.MAX_BREEDING_ATTEMPTS):
            
            offspring_1, offspring_2 = Cross_Over(self.population[individual_i], self.population[individual_j]).breed()
            
            if(offspring_1 is not None and offspring_2 is not None):
                
                mutated_offspring_1 = Mutation(offspring_1).mutate()
                mutated_offspring_2 = Mutation(offspring_2).mutate()

                # have to train the successfully bred graphs on the spot
                # because of tensorflow graphs handling problems :'(

                for offspring in [mutated_offspring_1, mutated_offspring_2]:
                    
                    if(successful_breedings < 2):
                            
                        try:
                            offspring_nn = NN(offspring)
                            offspring_fitness = self.train_network(offspring_nn)
                            self.next_generation_dna.append(offspring)
                            self.next_generation_nn.append(offspring_nn)
                            self.next_generation_fitness.append(offspring_fitness)
                            successful_breedings += 1

                        except Exception as e:
                            print("DNA ill-formed: Failed to build the NN \n\n" + str(e))
                        
                        tf.reset_default_graph()
                
            else:
                print("Failed to cross-over the individuals")
            
        return(successful_breedings)


    def train_network(self, nn):

        """
        Train the Neural Network for a fixed number of steps and regularly evaluate
        its performances 
        """

        best_validation_error = 1

        with tf.Session() as sess:
                
            # sess = tf.Session(graph=nn.get_graph())
            sess.run(tf.global_variables_initializer())
            # nn.sess.run(tf.initialize_local_variables())

            for training_step in range(Settings.TRAINING_STEPS):
                
                # training step
                images, labels = self.mnist.train.next_batch(Settings.MINIBATCH_SIZE)
                images = reshape_mnist_images(images)

                sess.run(nn.optimize, {nn.input: images, nn.labels: labels})

                # performance evaluation every few training steps
                if not training_step % Settings.EVALUATION_RATE:
                    validation_error = sess.run(nn.prediction_error, {nn.input: self.validation_data, nn.labels: self.validation_labels})
                    print('Test error {:6.2f}%'.format(100 * validation_error))
                    best_validation_error = min(best_validation_error, validation_error)

            # save graph logs
            log_path = "./tmp/log/" + str(self.generation) + "/" + "{:7.5}".format(best_validation_error).strip()
            writer = tf.summary.FileWriter(log_path, sess.graph)
            writer.close()

        return(best_validation_error)


    def create_initial_population(self):

        """
        Create an initial population consisting of *population_size*
        primitive neural networks with an input-output structure
        """

        for individual_i in range(self.population_size):
            
            individual_dna = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)
            individual_dna.create_primitive_structure()

            individual_nn = NN(individual_dna)
            individual_fitness = self.train_network(individual_nn)
            
            tf.reset_default_graph()

            self.population.append(individual_dna)
            self.neural_networks.append(individual_nn)
            self.fitness.append(individual_fitness)
