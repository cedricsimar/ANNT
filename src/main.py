
from settings import Settings

# initialize Settings class with json file
Settings.load("./settings.json")

from ga import GeneticAlgorithm

def main():

    GA = GeneticAlgorithm(Settings.POPULATION_SIZE)
    GA.evolve(Settings.MAX_GENERATIONS)




if __name__ == "__main__":
    main()