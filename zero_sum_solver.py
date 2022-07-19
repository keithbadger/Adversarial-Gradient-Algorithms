import numpy as np
import scipy.optimize as opt

index = 1 # index of given grid size
n = 2 # grid size
game_file = open(str(n)+'x'+str(n)+ " #" + str(index), 'r')
G = np.loadtxt(game_file)
game_file.close()


c = np.ones(n)
A = -1*(G.T - G.min()+1)
print(A)
b = -1*np.ones(n)
print(b)
x = opt.linprog(c, A_ub=A, b_ub=b)
x = np.array(x.x)
x = np.round(x,2)
print(x)








