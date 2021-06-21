import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':
    functions = ['Ackley', 'Rastrigin', 'Sphere', 'Rosenbrock', 'Michalewitz', 'Griewank', 'Schwefel', 'Sum_squares', 'Zakharov', 'Powell']
    y_lim = {'Ackley' : [0, 15], 'Rastrigin' : [0,200], 'Sphere': [0,10], 'Rosenbrock':[0,400], 'Michalewitz':[-30, -5], 'Griewank':[0,10],
                'Schwefel':[3000,10000], 'Sum_squares':[0,10], 'Zakharov':[0,80], 'Powell':[0, 150]}
    for f in functions:
        x1 = np.zeros(400000, dtype=float)
        x2 = np.zeros(400000, dtype=float)
        x3 = np.zeros(400000, dtype=float)
        t = pd.read_csv('./output/' + f, index_col=0, header=None)
        t = t.to_numpy()

        for i in range(len(t)):
            x1[i] = t[i][0]
        t = pd.read_csv('./output_BBC/' + f.lower(), index_col=0, header=None)
        t = t.to_numpy()
        for i in range(len(t)):
            x2[i] = t[i][0]
        t = pd.read_csv('./output_pso/' + f.lower(), index_col=0, header=None)
        t = t.to_numpy()
        for i in range(len(t)):
            x3[i] = t[i][0]
        x1 = x1[::100]
        x2 = x2[::100]
        x3 = x3[::100]
        print(x2)
        print(x3)
        x = [i*100 for i in range(len(x1))]
        plt.title(f, fontsize=38)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.ylim(y_lim[f][0], y_lim[f][1])
        plt.xlabel('Evaluations', fontsize=38)
        plt.ylabel('Fitness', fontsize=38)
        plt.grid(True)
        plt.plot(x, x1, label='AB3C', linewidth=6.0)
        plt.plot(x, x2, label='N_AB3C', linewidth=5.0)
        plt.plot(x, x3, label='PSO', linewidth=5.0)
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

