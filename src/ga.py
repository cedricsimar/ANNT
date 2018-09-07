
from settings import Settings
from exceptions import NoBridgeException, SaveMyLaptopException
from random import randint
from copy import deepcopy
import os
import pickle
import time
from datetime import datetime

from dna import DNA
from mutation import Mutation
from cross_over import Cross_Over
from nn import NN
from utils import reshape_mnist_images, reshape_cifar10_images, clean_folder, copy_files_from_to

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input
import cifar10 as cifar10_input

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class GeneticAlgorithm(object):

    def __init__(self, population_size, dataset='m'):

        self.generation = 0
        self.population_size = population_size
        self.mutations_per_breeding = Settings.MUTATIONS_PER_BREEDING
        self.dataset = dataset

        # Select dataset: m = MNIST, c = CIFAR-10.
        if(self.dataset == 'm'):
            self.data = mnist_input.read_data_sets("MNIST_data", one_hot=True,
                                                    validation_size=0)
        else:
            self.data = cifar10_input.read_data_sets(validation_size=0)

        self.population = []
        self.fitness = []

        self.best_ever_fitness = 1
        self.best_generations_fitness = []

        # Save elapsed times for each generation in a text file.
        self.gentimes = 'gentimes_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt'
        self.gentimes_writer = open(self.gentimes, 'w')

        # If there is a check point, load it, otherwise start with an initial population.
        if not os.path.exists(Settings.CHECKPOINT_PATH + self.dataset + "\\"):
            os.mkdir(Settings.CHECKPOINT_PATH + self.dataset + "\\")

        self.start_generation = len(os.listdir(Settings.CHECKPOINT_PATH + self.dataset + "\\")) - 1

        if(self.start_generation+1):
            self.load_last_saved_generation()
        else:
            beginTime = time.time()
            self.create_initial_population()
            endTime = time.time()
            print('TIME TAKEN FOR PRODUCING INITIAL POPULATION: {:5.2f}s\n'.format(endTime - beginTime))
            self.gentimes_writer.write(
                'Initial population time: ' + str(endTime - beginTime) + 's\n')


    def evolve(self, num_generations):

        """
        Evolve the population for a fixed number of generations
        """

        for gen in range(self.start_generation, num_generations):

            self.generation += 1
            print("\a-------------------------------------------------")
            print("EVOLVING GENERATION " + str(self.generation))
            print("-------------------------------------------------")
            beginTime = time.time()

            self.next_generation_dna = []
            self.next_generation_fitness = []

            # save the best individuals (N_BEST_CANDIDATES),
            # they go through the next generation unchanged.
            for best_i in range(Settings.N_BEST_CANDIDATES):
                self.next_generation_dna.append(self.population[best_i])
                self.next_generation_fitness.append(self.fitness[best_i])

            currTime = time.time()
            timeDif = currTime - beginTime

            print("\a-------------------------------------------------")
            print("BREEDING BEST INDIVIDUALS (GEN " + str(self.generation) + ")")
            print("-------------------------------------------------")
            # breed the best individuals together
            for best_i in range(Settings.N_BEST_CANDIDATES):
                for best_j in range(best_i + 1, Settings.N_BEST_CANDIDATES):

                    currTime = time.time()
                    # if the CHECK_TIME cycle just looped, write info.
                    if (currTime - beginTime) % Settings.CHECK_TIME < timeDif % Settings.CHECK_TIME:
                        self.gentimes_writer.write(
                            'Generation #' + str(self.generation) + ' time: '
                            + str(currTime - beginTime) + 's\tindividuals: '
                            + str(len(self.next_generation_dna)) + ' out of '
                            + str(self.population_size) + '\n')
                    timeDif = currTime - beginTime

                    print("\a==========================================")
                    print("Breeding individuals", best_i, "and", best_j,
                          "(out of", Settings.N_BEST_CANDIDATES, ")")
                    self.attempt_breeding(best_i, best_j)

            print("\a-------------------------------------------------")
            print("BREEDING BEST WITH RANDOM (GEN " + str(self.generation) + ")")
            print("-------------------------------------------------")
            # breed each best individual with a random individual from the population
            # including itself because, well.. parthenogenesis and all..
            for best_i in range(Settings.N_BEST_CANDIDATES):

                currTime = time.time()
                # if the CHECK_TIME cycle just looped, write info.
                if (currTime - beginTime) % Settings.CHECK_TIME < timeDif % Settings.CHECK_TIME:
                    self.gentimes_writer.write(
                        'Generation #' + str(self.generation) + ' time: '
                        + str(currTime - beginTime) + 's\tindividuals: '
                        + str(len(self.next_generation_dna)) + ' out of '
                        + str(self.population_size) + '\n')
                timeDif = currTime - beginTime

                print("\a==========================================")
                print("Breeding individual", best_i, "(out of",
                      Settings.N_BEST_CANDIDATES, ") with a RANDOM individual")
                other_i = randint(0, self.population_size - 1)
                self.attempt_breeding(best_i, other_i)

            print("\a-------------------------------------------------")
            print("BREEDING RANDOM INDIVIDUALS (GEN " + str(self.generation) + ")")
            print("-------------------------------------------------")
            # breed two different random individuals together until the number of offsprings
            # reaches the maximum population size

            print("Number of individuals already produced:",
                  len(self.next_generation_dna), "out of", self.population_size)

            while(len(self.next_generation_dna) < self.population_size):

                currTime = time.time()
                # if the CHECK_TIME cycle just looped, write info.
                if (currTime - beginTime) % Settings.CHECK_TIME < timeDif % Settings.CHECK_TIME:
                    self.gentimes_writer.write(
                        'Generation #' + str(self.generation) + ' time: '
                        + str(currTime - beginTime) + 's\tindividuals: '
                        + str(len(self.next_generation_dna)) + ' out of '
                        + str(self.population_size) + '\n')
                timeDif = currTime - beginTime
                # if PATIENCE_TIME has already elapsed, end this generation.
                if timeDif > Settings.PATIENCE_TIME:
                    break

                print("\a==========================================")
                print("Breeding random individuals together (",
                      self.population_size - len(self.next_generation_dna),
                      "random breedings to complete population)")
                other_i = randint(0, self.population_size - 1)
                other_j = randint(0, self.population_size - 1)
                self.attempt_breeding(other_i, other_j)

            # sort the next generation individuals by fitness
            self.next_generation_fitness, self.next_generation_dna = (list(t) for t in zip(*sorted(zip(self.next_generation_fitness, self.next_generation_dna))))

            # update the population for the next generation
            self.population = self.next_generation_dna
            self.fitness = self.next_generation_fitness

            # update the best_generations_fitness list
            self.best_generations_fitness.append(self.fitness[0])

            # update the number of mutations per breeding if necessary
            # the number of mutations can be increased temporarily to get out of a local optimum
            gen_i = len(self.best_generations_fitness) - 1
            previous_gen_i = gen_i - 1

            if(self.best_generations_fitness[previous_gen_i] <= self.best_generations_fitness[gen_i]):
                self.mutations_per_breeding += 1
            else:
                self.mutations_per_breeding = 1

            # save generation checkpoint
            #self.save_generation_checkpoint()
            currTime = time.time()

            print("\n ===============================================")
            print("\n ============== END OF GENERATION ==============")
            print("\nFitness history through generations :", self.best_generations_fitness)
            best_so_far = max(self.best_generations_fitness)
            print("The best individual so far is from generation", self.best_generations_fitness.index(best_so_far),
                  "with a fitness of ", best_so_far)
            print('TIME TAKEN FOR EXECUTING THIS GENERATION: {:5.2f}s\n'.format(currTime - beginTime))
            print("\n ===============================================")
            self.gentimes_writer.write(
                'Generation #' + str(self.generation) + ' time: '
                + str(currTime - beginTime) + 's\tindividuals: '
                + str(len(self.next_generation_dna)) + ' out of '
                + str(self.population_size) + '\n')


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
            print("==========================================")
            print("Succesful breedings: ", successful_breedings, "/ 2",
                  "\nBreeding attempts:", breeding_attempts, "/",
                  Settings.MAX_BREEDING_ATTEMPTS)

            try:
                # cross over the two individuals
                offspring_1, offspring_2 = Cross_Over(
                    self.population[individual_i],
                    self.population[individual_j]).breed(self.dataset)

            except NoBridgeException as e:
                print(e)
                print("Failed to cross-over the individuals")
                return (False)
            except Exception as e:
                print("Failed to cross-over the individuals")
                return (False)


            for offspring in [offspring_1, offspring_2]:

                # apply (several if we are stuck in a local optimum) mutations to the offspring
                for _ in range(self.mutations_per_breeding):
                    offspring = Mutation(offspring).mutate()

                mutated_offspring = offspring

                # build and train the crossed-mutated graphs on the spot
                if(successful_breedings < 2):

                    try:
                        mutated_offspring_fitness = self.build_and_train_network(mutated_offspring)

                        self.next_generation_dna.append(mutated_offspring)
                        self.next_generation_fitness.append(mutated_offspring_fitness)
                        successful_breedings += 1

                    except Exception as e:
                        print("Failed to build the NN \n\n" + str(e))

            breeding_attempts += 1

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
            num_trainable_variables = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
            print("==========================================")
            print("Number of trainable variables :", num_trainable_variables)
            print("Training the following DNA: \n")
            print(nn.dna)

            if(num_trainable_variables > Settings.MAX_TRAINABLE_PARAMETERS):
                raise SaveMyLaptopException()

            # Configure GPU options to avoid memory fragmentation.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            #config.gpu_options.per_process_gpu_memory_fraction = 0.3

            with tf.Session(graph=graph, config=config) as sess:

                # merge the summaries and create a file writer in the tmp folder
                clean_folder(Settings.TMP_FOLDER_PATH + self.dataset + "\\")
                merged_summaries = tf.summary.merge_all()
                writer = tf.summary.FileWriter(Settings.TMP_FOLDER_PATH + self.dataset + "\\", sess.graph)

                # initialize variables and finalize the graph
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

                for training_step in range(Settings.TRAINING_STEPS):

                    # training step
                    images, labels = self.data.train.next_batch(Settings.MINIBATCH_SIZE)
                    if(self.dataset == 'm'):
                        images = reshape_mnist_images(images)
                    else:
                        images = reshape_cifar10_images(images)

                    sess.run(nn.optimize, {nn.input: images, nn.labels: labels, nn.is_training: True})

                    # performance evaluation every few training steps
                    if not training_step % Settings.EVALUATION_RATE:

                        test_images, test_labels = self.data.test.next_batch(Settings.TEST_BATCH_SIZE)
                        if(self.dataset == 'm'):
                            test_images = reshape_mnist_images(test_images)
                        else:
                            test_images = reshape_cifar10_images(test_images)

                        summary, validation_error = sess.run([merged_summaries, nn.prediction_error], {nn.input: test_images, nn.labels: test_labels, nn.is_training: False})
                        writer.add_summary(summary, training_step)
                        print('Test error {:6.2f}%'.format(100 * validation_error))
                        best_validation_error = min(best_validation_error, validation_error)

                # close the writer
                writer.close()

                # save graph logs and dna string into the right models folder
                model_path = Settings.MODELS_FOLDER_PATH + self.dataset + "\\" + str(self.generation) + "\\" + "{:f}".format(best_validation_error)

                copy_files_from_to(Settings.TMP_FOLDER_PATH + self.dataset + "\\", model_path)

                file_path = model_path + "\\topology.txt"
                with open(file_path, 'a') as topo_file:
                    topo_file.write(nn.dna.__str__())

        print("Training and Saving complete !\n\n")

        return(best_validation_error)


    def create_initial_population(self):

        """
        Create an initial population consisting of *population_size*
        primitive neural networks with an input-output structure
        """

        for individual_i in range(self.population_size):
            print("==========================================")
            print("Creating individual", individual_i, "out of", self.population_size)

            if(self.dataset == 'm'):
                individual_dna = DNA(Settings.INPUT_SHAPE_MNIST, Settings.OUTPUT_SHAPE)
            else:
                individual_dna = DNA(Settings.INPUT_SHAPE_CIFAR10, Settings.OUTPUT_SHAPE)
            individual_dna.create_primitive_structure()

            individual_fitness = self.build_and_train_network(individual_dna)

            self.population.append(individual_dna)
            self.fitness.append(individual_fitness)

        # sort these first individuals by fitness
        self.fitness, self.population = (list(t) for t in zip(*sorted(zip(self.fitness, self.population))))

        # update the best_generations_fitness list
        self.best_generations_fitness.append(self.fitness[0])

        # save a checkpoint for the initial population
        self.save_generation_checkpoint()


    def load_last_saved_generation(self):

        print("Loading generation", self.start_generation, "from checkpoint..  ", end='')

        last_checkpoint_path = Settings.CHECKPOINT_PATH + self.dataset + "\\" + str(self.start_generation) + "\\"

        # load the population list
        population_file = open(last_checkpoint_path + "population.pickle", "rb")
        self.population = pickle.load(population_file)
        population_file.close()

        # load the fitness list
        fitness_file = open(last_checkpoint_path + "fitness.pickle", "rb")
        self.fitness = pickle.load(fitness_file)
        fitness_file.close()

        # load the best generations fitness list
        best_fitness_file = open(last_checkpoint_path + "best_fitness.pickle", "rb")
        self.best_generations_fitness = pickle.load(best_fitness_file)
        best_fitness_file.close()

        # restore generation number
        self.generation = self.start_generation

        print("done.")


    def save_generation_checkpoint(self):

        save_path = Settings.CHECKPOINT_PATH + self.dataset + "\\" + str(self.generation) + "\\"
        os.mkdir(save_path)

        print("Saving generation", self.generation, "in the checkpoint folder..  ", end='')

        # save the population list
        population_file = open(save_path + "population.pickle", "wb")
        pickle.dump(self.population, population_file)
        population_file.close()

        # save the fitness list
        fitness_file = open(save_path + "fitness.pickle", "wb")
        pickle.dump(self.fitness, fitness_file)
        fitness_file.close()

        # save the best generations fitness list
        best_fitness_file = open(save_path + "best_fitness.pickle", "wb")
        pickle.dump(self.best_generations_fitness, best_fitness_file)
        best_fitness_file.close()

        print("done.")
