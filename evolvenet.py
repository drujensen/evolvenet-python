import logging

from neuralnetwork import NeuralNetwork
from organism import Organism
from utils import confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    nn = NeuralNetwork()
    nn.add_layer("input", size=2)
    nn.add_layer("hidden", size=4)
    nn.add_layer("output", size=1)
    nn.fully_connect()

    data = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    organism = Organism(nn)
    network = organism.evolve(data, generations=1000)

    print(network.to_json())

    logging.info(f"Final error: {network.error}")
    confusion_matrix(network, data)


if __name__ == "__main__":
    main()
