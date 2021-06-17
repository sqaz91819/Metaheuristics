import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':
    functions = ['Ackley', 'Rastrigin', 'Sphere', 'Rosenbrock', 'Michalewitz']
    for f in functions:
        x1 = np.zeros(4000, dtype=float)
        x2 = np.zeros(4000, dtype=float)
        x3 = np.zeros(4000, dtype=float)
        t = pd.read_csv('./output/' + f, index_col=0, header=None)
        t = t.to_numpy()

        for i in range(len(t)):
            x1[i] = t[i][0]
        t = pd.read_csv('./output1/' + f.lower(), index_col=0, header=None)
        t = t.to_numpy()
        for i in range(len(t)):
            x2[i] = t[i][0]
        t = pd.read_csv('./output_pso/' + f.lower(), index_col=0, header=None)
        t = t.to_numpy()
        for i in range(len(t)):
            x3[i] = t[i][0]
        x = [i for i in range(4000)]
        plt.title(f, fontsize=38)
        plt.xticks(fontsize=38)
        plt.yticks(fontsize=38)
        plt.xlabel('Iteration', fontsize=38)
        plt.ylabel('Fitness', fontsize=38)
        plt.grid(True)
        plt.plot(x, x1, label='AB3C', linewidth=6.0)
        plt.plot(x, x2, label='SEMSO', linewidth=6.0)
        plt.plot(x, x3, label='PSO', linewidth=6.0)
        plt.legend(fontsize=38)
        
        plt.show()

    # x = np.zeros(4000, dtype=float)
    # t = pd.read_csv('./output1/Rastrigin', index_col=0, header=None)
    # t = t.to_numpy()
    # print(t)
    # for i in range(len(t)):
    #     x[i] = t[i][0]

    # x1 = [i for i in range(4000)]
    # print(x)
    # plt.plot(x1, x)
    # plt.show()

