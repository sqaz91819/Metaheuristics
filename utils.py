import math
from math import sin
from math import cos
from math import exp
from math import sqrt
from math import e

DOMAIN = {'Ackley' : [-30.0, 30.0], 'Rastrigin' : [-5.12, 5.12], 'Sphere' : [-100.0, 100.0],
                'Rosenbrock' : [-10.0, 10.0], 'Michalewitz' : [0.0, math.pi]}

def ackley(dim: int, coordination: list) -> float:
    '''
    return the value from ackley function
    '''
    s1, s2 = 0.0, 0.0
    for i in range(dim):
        s1 += coordination[i]**2
        s2 += cos(2.0 * math.pi * coordination[i])

    return -20.0 * exp(-0.2 * sqrt(s1 / dim)) - exp(s2 / dim) + 20.0 + e

def rastrigin(dim: int, coordination: list) -> float:
    if dim > 20:
        dim = 20

    sum = 0.0
    for i in range(dim):
        sum += coordination[i]**2 - cos(2.0 * math.pi * coordination[i])

    return sum + dim * 10

def sphere(dim: int, coordination: list) -> float:
    sum = 0.0
    for i in range(dim):
        sum += coordination[i]**2
    
    return sum

def rosenbrock(dim: int, coordination: list) -> float:
    s = 0.0
    for i in range(1, dim):
        s += 100.0 * ( (coordination[i] - coordination[i - 1]**2)**2) + (coordination[i-1] - 1.0)**2
    
    return s

def michalewitz(dim: int, coordination: list) -> float:
    u = 0.0
    for i in range(dim):
        u += sin(coordination[i]) * sin((i + 1) * coordination[i]**2 / math.pi)**(2.0 * 10.0)
    
    return -u

if __name__=='__main__':
    out = ackley(2, [0.0000001, 0.0000001])
    print(out)