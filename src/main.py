
from settings import Settings

# initialize Settings class with json file
Settings.load(".\\settings.json")

from ga import GeneticAlgorithm

def main():

    print("----------------------------------------------------")
    print("-------------------INITIALIZATION-------------------")
    print("----------------------------------------------------\n")
    GA = GeneticAlgorithm(Settings.POPULATION_SIZE, dataset='c')

    print("\a----------------------------------------------------")
    print("---------------EVOLUTIONARY ALGORITHM---------------")
    print("----------------------------------------------------\n")
    GA.evolve(3)
    GA.gentimes_writer.close()

if __name__ == "__main__":
    main()
