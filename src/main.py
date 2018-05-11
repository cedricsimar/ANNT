
from settings import Settings

# initialize Settings class with json file
Settings.load("./settings.json")

from ga import GeneticAlgorithm

def main():

    GA = GeneticAlgorithm(20, 1)
    GA.evolve(1)




if __name__ == "__main__":
    main()