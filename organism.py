import logging
import threading
from network import Network


class Organism:
    def __init__(self, network: Network, size: int = 16):
        if size < 16:
            raise ValueError("size needs to be greater than 16")
        self.networks = [network.clone().randomize() for _ in range(size)]
        one_forth = int(size * 0.25)
        self.one_forth = one_forth
        self.two_forth = one_forth * 2
        self.three_forth = one_forth * 3
        self.logger = logging.getLogger(self.__class__.__name__)

    def evolve(self,
               data,
               generations: int = 10000,
               error_threshold: float = 0.0,
               log_each: int = 1000):

        def worker(n, channel):
            n.evaluate(data)
            channel.append(n.error)

        for gen in range(generations + 1):
            channel = []
            threads = []
            for n in self.networks:
                thread = threading.Thread(target=worker, args=(n, channel))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            self.networks.sort(key=lambda x: x.error)

            error = self.networks[0].error
            if error <= error_threshold:
                self.logger.info(f"generation: {gen} error: {error}. below threshold. breaking.")
                break
            elif gen % log_each == 0:
                self.logger.info(f"generation: {gen} error: {error}")

            self.networks = self.networks[:self.three_forth]
            top_quarter = self.networks[:self.one_forth]
            for n in top_quarter:
                self.networks.append(n.clone())

            for i, n in enumerate(self.networks[1:4]):
                n.punctuate(i)

            for n in self.networks[4:]:
                n.mutate()

        self.networks.sort(key=lambda x: x.error)
        return self.networks[0]

    def to_json(self):
        return [network.to_json() for network in self.networks]
