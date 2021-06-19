import math
from math import sin
from math import cos
from math import exp
from math import sqrt
from math import e

DOMAIN = {'Ackley' : [-32.768, 32.768], 'Rastrigin' : [-5.12, 5.12], 'Sphere' : [-30, 30],
                'Rosenbrock' : [-10.0, 10.0], 'Michalewitz' : [0.0, math.pi], 'Griewank' : [-600, 600],
                'Schwefel' : [-500, 500], 'Sum_squares' : [-10, 10], 'Zakharov' : [-5, 10],
                'Powell' : [-4, 5]
        }

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
    sum = 0.0
    for i in range(dim):
        sum += coordination[i]**2 - 10 * cos(2.0 * math.pi * coordination[i])

    return sum + float(dim) * 10.0

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
        u += sin(coordination[i]) * sin(float(i + 1)* coordination[i]**2 / math.pi)**(2.0 * 10.0)
    
    return -1.0 * u

def griewank(dim: int, coordination: list) -> float:
    sum = 0.0
    prod = 1
    for i, xi in zip(range(1, dim+1), coordination):
        sum += xi**2 / 4000.0
        prod = prod * cos(xi/sqrt(i))
    
    return sum - prod + 1

def schwefel(dim: int, coordination: list) -> float:
    temp = sum([xi*sin(sqrt(abs(xi))) for xi in coordination])
    return 418.9829*dim - temp

def sum_squares(dim: int, coordination: list) -> float:
    temp = [i*(xi**2) for i, xi in zip(range(1, dim+1), coordination)]
    return sum(temp)

def zakharov(dim: int, coordination: list) -> float:
    temp_1 = [xi**2 for xi in coordination]
    temp_2 = [0.5*i*xi for i, xi in zip(range(1, dim+1), coordination)]
    return sum(temp_1) + sum(temp_2) ** 2 + sum(temp_2) ** 4

def powell(dim: int, coordination: list) -> float:
    sum = 0.0
    for i in range(1, int(dim/4)+1):
        term_1 = (coordination[4*i-3-1] + 10 * coordination[4*i-2-1])**2
        term_2 = 5 * (coordination[4*i-1-1] - coordination[4*i-1])**2
        term_3 = (coordination[4*i-2-1]- 2 * coordination[4*i-1-1])**4
        term_4 = 10 * (coordination[4*i-3-1] - coordination[4*i-1])**4
        sum = sum + term_1 + term_2 + term_3 + term_4
    return sum

if __name__=='__main__':
    # out = ackley(2, [0.0000001, 0.0000001])
    # out = sphere(4, [1.0,1.0,10.0, 1.0])
    # out = powell(4, [0.0,0.0,0.0,0.0])
    # out = zakharov(4, [0.0,0.0,0.0,0.0])
    # out = sum_squares(4, [0.0,0.0,0.0,0.0])
    # out = schwefel(1, [420.9687])
    # out = griewank(4, [0.0,0.0,0.0,0.0])
    # print(out)
    exit()