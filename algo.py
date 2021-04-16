'''
python ver = '3.8.3'
'''
import random
from utils import DOMAIN
from utils import ackley
import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self, lower: float=0.0, upper: float=1.0, dim: int=2):
        self.dim = dim
        self.coordination = [random.uniform(lower, upper) for _ in range(self.dim)]
        self.best = 9999

    # def rand_pos(self, upper, lower):
    #     self.coordination = [random.uniform(lower, upper) for _ in range(self.dim)]


# 1. Initialize the centra point, max movement distance, discount factor, reward factor
# 2. Set some points around the central point [-1,1]
# 3. p_e = p_e + the distance between center and edge point * (M_MAX * discount factor * reward factor)
# 4. check current best and best so far, converge to the best point(center) by n steps.
# 5. repeat step 2.
class MyAlgo:

    def __init__(self, dim: int=2, discount_factor: float=0.996, reward_factor: float=1.003,
            testing: str='Ackley'):
        self.run = 3
        self.dim = dim
        self.popsize = 100
        self.population = []
        self.discount_factor = discount_factor
        self.reward_factor = reward_factor
        self.iteration = 4000
        self.testing = testing
        self.lower_bound = DOMAIN[self.testing][0]
        self.upper_bound = DOMAIN[self.testing][1]
        self.max_mov = (self.upper_bound - self.lower_bound) * 0.6

    def push(self, center) -> None:
        for i in range(self.popsize):
            for d in range(self.dim):
                self.population[i].coordination[d] = self.population[i].coordination[d] * self.max_mov + center.coordination[d]
                if self.population[i].coordination[d] > self.upper_bound:
                    self.population[i].coordination[d] += (center.coordination[d] - self.population[i].coordination[d])*2
                elif self.population[i].coordination[d] < self.lower_bound:
                    self.population[i].coordination[d] += (center.coordination[d] - self.population[i].coordination[d])*2

    def pull(self, center) -> None:
        for i in range(self.popsize):
            for d in range(self.dim):
                pull_distance = random.uniform(center.coordination[d], self.population[i].coordination[d])
                # pull from the point to the center point between half and center point range[p+c/2, c]
                self.population[i].coordination[d] = pull_distance

    def pull_2(self, center) -> None:
        for i in range(self.popsize):
            for d in range(self.dim):
                v = center.coordination[d] - self.population[i].coordination[d]
                # pull from the point to the center point between half and center point range[p+c/2, c]
                self.population[i].coordination[d] += v / 3

    def algo(self):
        current_best = np.zeros(self.iteration, dtype=float)
        center = Particle(self.lower_bound, self.upper_bound, dim=self.dim)
        center.best = ackley(self.dim, center.coordination)
        for k in range(self.iteration):
            # 2. Set some points around the center point [-1,1]
            if k % 5 == 0:
                self.population = [Particle(-1.0, 1.0, self.dim) for _ in range(self.popsize)]
                self.push(center)
            else:
                # pull
                self.pull(center)

            x = np.zeros(self.popsize)
            y = np.zeros(self.popsize)

            # evaluate
            for i in range(self.popsize):
                self.population[i].best = ackley(self.dim, self.population[i].coordination)
                if self.population[i].best < center.best:
                    center = self.population[i]
                    self.max_mov = self.max_mov * self.reward_factor
                x[i] = self.population[i].coordination[2]
                y[i] = self.population[i].coordination[3]

            print(center.best)
            current_best[k] = center.best
            self.max_mov = self.max_mov * self.discount_factor

            # fig, ax = plt.subplots()
            # plt.axis([self.lower_bound, self.upper_bound, self.lower_bound, self.upper_bound])
            # plt.plot(x, y, 'o')
            # plt.show()

        return current_best

    def run_algo(self):
        recorder = np.zeros((self.run, self.iteration), dtype=float)
        # print(recorder)
        # print(recorder.shape)
        for i in range(self.run):
            algo = MyAlgo(dim=self.dim, testing=self.testing)
            recorder[i] = algo.algo()

        # print(recorder)
        x = np.sum(recorder, axis=0)
        print(x)

        x1 = np.arange(self.iteration)
        plt.plot(x1, x)
        plt.show()



if __name__=='__main__':
    algo = MyAlgo(dim=30, testing='Sphere')
    # algo.algo()
    algo.run_algo()
