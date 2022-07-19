import numpy as np
import matplotlib.pyplot as plt

index = 1  # index of given grid size
n = 2 # n is the number of rows and columns
l = 10000 # l is the number of iterations
alpha = .01 #stepsize parameter
e = 1 # penalty coeff
#scale = 1

x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
r_t = np.array([]) # record of rewards
game_file = open(str(n)+'x'+str(n)+ " #" + str(index), 'r')
G = np.loadtxt(game_file)
game_file.close()

for k in range(l):
    #alpha = scale/(scale + k)
    #e = scale/(100*scale + k)
    r_t = np.append(r_t, x.transpose() @ G @ y)
    xbar = x * np.exp(e * G @ y)
    xbar = xbar/xbar.sum()
    ybar = y * np.exp(-e * G.T @ x)
    ybar = ybar/ybar.sum()
    x = x + alpha*(xbar - x)
    y = y + alpha*(ybar - y)

disp = np.zeros((n,2))
for p in range(n):
    disp[p,0] = round(x[p][0],2)
    disp[p,1] = round(y[p][0],2)
print(disp)
#"""
t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()
