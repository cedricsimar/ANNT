
from settings import Settings
from exceptions import NoBridgeException
from random import randint
from copy import deepcopy

from dna import DNA
from mutation import Mutation
from cross_over import Cross_Over
from nn import NN
from utils import reshape_mnist_images

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GeneticAlgorithm(object):

    def __init__(self, population_size):
        
        self.generation = 0
        self.population_size = population_size
        self.mutations_per_breeding = Settings.MUTATIONS_PER_BREEDING

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        
        self.validation_data = reshape_mnist_images(self.mnist.test.images)
        self.validation_labels = self.mnist.test.labels

        self.population = []
        self.fitness = []
        
        self.best_ever_fitness = 1
        self.best_generations_fitness = []

        self.create_initial_population()


    def evolve(self, num_generations):
        
        """
        Evolve the population for a fixed number of generations
        """

        for gen in range(num_generations):

            self.generation += 1
            print("Evolving generation " + str(self.generation))

            self.next_generation_dna = []
            self.next_generation_fitness = []

            # save the three best individuals that go through the next generation unchanged
            # (produces N_BEST_CANDIDATES offsprings)
            for best_i in range(Settings.N_BEST_CANDIDATES):

                self.next_generation_dna.append(self.population[best_i])
                self.next_generation_fitness.append(self.fitness[best_i])

            # breed the three best individuals together
            # (produces at most 6 offsprings)
            for best_i in range(Settings.N_BEST_CANDIDATES):
                for best_j in range(best_i + 1, Settings.N_BEST_CANDIDATES):
                    self.attempt_breeding(best_i, best_j)
            
            # breed each best individual with a random individual from the population
            # including itself because, well.. parthenogenesis and all..
            # (produces at most 6 offsprings)
            for best_i in range(Settings.N_BEST_CANDIDATES):
                other_i = randint(0, self.population_size - 1)
                self.attempt_breeding(best_i, other_i)

            # breed two different random individuals together until the number of offsprings 
            # reaches the maximum population size
            while(len(self.next_generation_dna) < self.population_size):
                other_i = randint(0, self.population_size - 1)
                other_j = randint(0, self.population_size - 1)
                self.attempt_breeding(other_i, other_j)
            
            # sort the next generation individuals by fitness
            self.next_generation_fitness, self.next_generation_dna = (list(t) for t in zip(*sorted(zip(self.next_generation_fitness, self.next_generation_dna))))

            # update the population for the next generation
            self.population = self.next_generation_dna
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
            
            try:
                offspring_1, offspring_2 = Cross_Over(self.population[individual_i], self.population[individual_j]).breed()
            
            except NoBridgeException as e:
                print(e)
                print("Failed to cross-over the individuals")
                return (False)
                

            for offspring in [offspring_1, offspring_2]:
            
                mutated_offspring = Mutation(offspring).mutate()

                # Build and train the crossed-mutated graphs on the spot
                if(successful_breedings < 2):
                        
                    try:
                        mutated_offspring_fitness = self.build_and_train_network(mutated_offspring)

                        self.next_generation_dna.append(mutated_offspring)
                        self.next_generation_fitness.append(mutated_offspring_fitness)
                        successful_breedings += 1

                    except Exception as e:
                        print("DNA ill-formed: Failed to build the NN \n\n" + str(e))
                    
          
        return(successful_breedings)


    def build_and_train_network(self, individual_dna):

        """
        Train the Neural Network for a fixed number of steps and regularly evaluate
        its performances 
        """

        best_validation_error = 1

        graph = tf.Graph()
        with graph.as_default():
                
            # build the Neural Network using the mutated dna instance
            nn = NN(individual_dna)

            # and train the resulting model


            print("==========================================")
            print("Number of trainable variables : ", end='')
            print(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
            print("Training the following DNA: \n")
            print(nn.dna)

            with tf.Session(graph=graph) as sess:
                    
                # sess = tf.Session(graph=nn.get_graph())
                sess.run(tf.global_variables_initializer())
                # nn.sess.run(tf.initialize_local_variables())
                sess.graph.finalize()

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

                # save graph logs and dna string
                log_path = "./tmp/log/" + str(self.generation) + "/" + "{:f}".format(best_validation_error)
                
                writer = tf.summary.FileWriter(log_path, sess.graph)
                writer.close()

                file_path = log_path + "/topology.txt"
                with open(file_path, 'w') as topo_file:
                    topo_file.write(nn.dna.__str__())
         
        print("Training and Saving complete !")

        return(best_validation_error)


    def create_initial_population(self):

        """
        Create an initial population consisting of *population_size*
        primitive neural networks with an input-output structure
        """

        for individual_i in range(self.population_size):
            
            individual_dna = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)
            individual_dna.create_primitive_structure()

            individual_fitness = self.build_and_train_network(individual_dna)

            self.population.append(individual_dna)
            self.fitness.append(individual_fitness)
