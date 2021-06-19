'''
python ver = '3.8.3'
'''
import random
from utils import DOMAIN
from utils import ackley
from utils import rosenbrock
from utils import sphere
from utils import michalewitz
from utils import rastrigin
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threading
import time
import multiprocessing as mp

class Particle:
    def __init__(self, lower: float=0.0, upper: float=1.0, dim: int=2):
        self.dim = dim
        self.coordination = [random.uniform(lower, upper) for _ in range(self.dim)]
        self.best = 9999

    # def rand_pos(self, upper, lower):
    #     self.coordination = [random.uniform(lower, upper) for _ in range(self.dim)]

class PSO_Particle(Particle):
    def __init__(self, lower: float, upper: float, dim: int):
        super().__init__(lower=lower, upper=upper, dim=dim)
        self.personal_v = [random.uniform(-10, 10) for _ in range(self.dim)]
        self.personal_c = self.coordination # personal coordination
        self.current = self.best

    def init_particle(self, fitness):
        self.best = fitness
        self.current = self.best


class Algo:
    def __init__(self, dim: int=2, testing: str='Ackley'):
        self.run = 30
        self.dim = dim
        self.testing = testing
        self.lower_bound = DOMAIN[self.testing][0]
        self.upper_bound = DOMAIN[self.testing][1]
        self.popsize = 100
        self.iteration = 4000
        self.evaluation = self.iteration * self.popsize

    def algo(self):
        raise NotImplementedError()

class PSO(Algo):
    def __init__(self, dim: int=2, testing: str='Ackley'):
        super(PSO, self).__init__(dim, testing)
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.delta = 0
        # self.popsize = 30

    def algo(self):
        best_so_far = 16777216
        current_best = np.zeros(self.evaluation, dtype=float)
        current_evl = 0
        Particles = [PSO_Particle(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.popsize)]
        for p in Particles:
            p.init_particle(globals()[self.testing.lower()](self.dim, p.coordination))
        
        best_p = Particles[0]
        while current_evl < self.evaluation:
            for p in Particles:
                if p.current < best_p.current:
                    best_p = p

            for p in Particles:
                for d in range(self.dim):
                    lo1 = random.random()
                    lo2 = random.random()
                    p.personal_v[d] = 0.7 * p.personal_v[d] + lo1 * 1.496180 * (p.personal_c[d] - p.coordination[d]) + lo2 * 1.496180 * (best_p.coordination[d] - p.coordination[d])
                    if p.personal_v[d] > 1.0:
                        p.personal_v[d] = 1.0
                    if p.personal_v[d] < -1.0:
                        p.personal_v[d] = -1.0

                for d in range(self.dim):
                    p.coordination[d] += p.personal_v[d]
                    if p.coordination[d] < self.lower_bound:
                        p.coordination[d] = self.lower_bound
                    if p.coordination[d] > self.upper_bound:
                        p.coordination[d] = self.upper_bound

                p.current = globals()[self.testing.lower()](self.dim, p.coordination)

                if p.current < p.best:
                    p.best = p.current
                    p.personal_c = p.coordination

                if p.current < best_p.current:
                    best_p = p

                current_evl += 1
            print(best_p.current)


    def my_run(self):
        print('?')
        self.algo()

# 1. Initialize the centra point, max movement distance, discount factor, reward factor
# 2. Set some points around the central point [-1,1]
# 3. p_e = p_e + the distance between center and edge point * (M_MAX * discount factor * reward factor)
# 4. check current best and best so far, converge to the best point(center) by n steps.
# 5. repeat step 2.
class MyAlgo(Algo):

    def __init__(self, dim: int=2, discount_factor: float=0.9965, reward_factor: float=1.009,
            testing: str='Ackley'):
        super(MyAlgo, self).__init__(dim, testing)
        self.run = 1
        self.population = []
        self.discount_factor = discount_factor
        self.reward_factor = reward_factor
        self.max_mov = abs(self.upper_bound - self.lower_bound) * 0.5

    def push(self, center) -> None:
        for i in range(len(self.population)):
            for d in range(self.dim):
                self.population[i].coordination[d] = self.population[i].coordination[d] * self.max_mov + center.coordination[d]
                if self.population[i].coordination[d] > self.upper_bound:
                    self.population[i].coordination[d] += (center.coordination[d] - self.population[i].coordination[d])*2
                elif self.population[i].coordination[d] < self.lower_bound:
                    self.population[i].coordination[d] += (center.coordination[d] - self.population[i].coordination[d])*2

    def pull(self, center) -> None:
        for i in range(len(self.population)):
            for d in range(self.dim):
                pull_distance = random.uniform(center.coordination[d], self.population[i].coordination[d])
                # pull from the point to the center point between half and center point range[p+c/2, c]
                self.population[i].coordination[d] = pull_distance

    def pull_2(self, center) -> None:
        for i in range(len(self.population)):
            for d in range(self.dim):
                v = center.coordination[d] - self.population[i].coordination[d]
                # pull from the point to the center point between half and center point range[p+c/2, c]
                self.population[i].coordination[d] += v / 3

    def algo(self):
        best_so_far = 16777216
        current_best = np.zeros(self.evaluation, dtype=float)
        center = Particle(self.lower_bound, self.upper_bound, dim=self.dim)
        # center2 = Particle(self.lower_bound, self.upper_bound, dim=self.dim)
        center.best = globals()[self.testing.lower()](self.dim, center.coordination)
        # center2.best = ackley(self.dim, center.coordination)
        current_evl = 0
        k=0
        while current_evl < self.evaluation:
            # 2. Set some points around the center point [-1,1]
            if k % 10 == 0:
                self.population = [Particle(-1.0, 1.0, self.dim) for _ in range(self.popsize)]
                self.push(center)
            else:
                # pull
                self.pull(center)

            x = np.zeros(len(self.population))
            y = np.zeros(len(self.population))

            # evaluate
            for i in range(len(self.population)):
                # self.population[i].best = ackley(self.dim, self.population[i].coordination)
                self.population[i].best = globals()[self.testing.lower()](self.dim, self.population[i].coordination)
                if self.population[i].best <= center.best:
                    center = self.population[i]
                    self.max_mov = self.max_mov * self.reward_factor
                    if random.random() < 0.5:
                        self.popsize += 1
                x[i] = self.population[i].coordination[0]
                y[i] = self.population[i].coordination[1]
                if current_evl < self.evaluation:
                    current_best[current_evl] = best_so_far
                current_evl += 1
            if center.best < best_so_far:
                best_so_far = center.best
                self.popsize += 2

            if random.random() < 0.5:
                self.popsize = self.popsize - 1
            print(best_so_far)
            # print('size :', self.popsize)
            self.max_mov = self.max_mov * self.discount_factor
            k = k + 1

            if self.popsize == 0:
                center = Particle(self.lower_bound, self.upper_bound, dim=self.dim)
                center.best = globals()[self.testing.lower()](self.dim, center.coordination)
                self.popsize = 100
                if k % 10 != 0:
                    self.population = [Particle(-1.0, 1.0, self.dim) for _ in range(self.popsize)]


            # fig, ax = plt.subplots()
            # plt.axis([self.lower_bound, self.upper_bound, self.lower_bound, self.upper_bound])
            # plt.plot(x, y, 'o')
            # plt.show()

        print(self.testing , best_so_far)
        return current_best

    def run_algo(self):
        recorder = np.zeros((self.run, self.evaluation), dtype=float)
        # print(recorder)
        # print(recorder.shape)
        for i in range(self.run):
            algo = MyAlgo(dim=self.dim, testing=self.testing)
            recorder[i] = algo.algo()
            print(self.testing, 'run :', i)

        # print(recorder)
        # x = np.sum(recorder, axis=0)
        x = np.mean(recorder, axis=0)
        # print(x)

        # x1 = np.arange(self.iteration)
        # plt.plot(x1, x)
        # plt.show()
        pd.DataFrame(x).to_csv("./output/" + self.testing, header=None)



if __name__=='__main__':
    t = time.time()
    test = False
    if not test:
        functions = ['Ackley', 'Rastrigin', 'Sphere', 'Rosenbrock', 'Michalewitz', 'Griewank', 'Schwefel', 'Sum_squares', 'Zakharov', 'Powell']
        threads = []
        tasks = [MyAlgo(dim=30, testing=function) for function in functions]
        for i in range(len(tasks)):
            threads.append(mp.Process(target=tasks[i].run_algo))
            threads[i].start()
        for i in range(len(tasks)):
            threads[i].join()

    # for function in functions:
    #     algo = MyAlgo(dim=30, testing=function)
    #     algo.run_algo()
    # algo.algo()
    # algo.run_algo()

    # algo = MyAlgo(dim=30, testing='Schwefel')
    # algo.run_algo()
    else:
        pso = PSO(dim=30, testing='Sphere')
        pso.my_run()

    print(time.time() - t, 'sec')
    exit()
