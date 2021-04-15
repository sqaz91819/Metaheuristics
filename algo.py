'''
python ver = '3.8.3'
'''
import random
from utils import DOMAIN
from utils import ackley

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

    def __init__(self, dim: int=2, discount_factor: float=0.99, reward_factor: float=1.01,
            max_mov: float=20.0, testing: str='Ackley'):
        self.dim = dim
        self.popsize = 50
        self.population = []
        self.discount_factor = discount_factor
        self.reward_factor = reward_factor
        self.max_mov = max_mov
        self.iteration = 1
        self.testing = testing
        self.lower_bound = DOMAIN[self.testing][0]
        self.upper_bound = DOMAIN[self.testing][1]

    def push(self, center) -> None:
        for i in range(self.popsize):
            for d in range(self.dim):
                self.population[i].coordination[d] = self.population[i].coordination[d] * self.max_mov + center.coordination[d]

    def pull(self, center) -> None:
        for i in range(self.popsize):
            for d in range(self.dim):
                pull_distance = random.uniform(center.coordination[d], (self.population[i].coordination[d] - center.coordination[d]) * 3 / 4)
                # pull from the point to the center point between half and center point range[p+c/2, c]
                self.population[i].coordination[d] = (self.population[i].coordination[d] + center.coordination[d]) / pull_distance

    def algo(self):
        center = Particle(self.lower_bound, self.upper_bound, dim=self.dim)
        center.best = ackley(self.dim, center.coordination)
        for _ in range(self.iteration):
            # 2. Set some points around the center point [-1,1]
            self.population = [Particle(-1.0, 1.0, self.dim) for i in range(self.popsize-1)]
            self.population.append(center)
            self.push(center)
            # pull
            self.pull(center)

            # evaluate
            for i in range(self.popsize):
                self.population[i].best = ackley(self.dim, self.population[i].coordination)
                if self.population[i].best < center.best:
                    center = self.population[i]
