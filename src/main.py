
from settings import Settings

# initialize Settings class with json file
Settings.load("./settings.json")

from ga import GeneticAlgorithm

# add max pooling, dropout, batch normalization

def main():

    GA = GeneticAlgorithm(10, 1)




if __name__ == "__main__":
    main()