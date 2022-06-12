import numpy as np
import matplotlib.pyplot as plt
n = 20 # n is the number of rows and columns
l = 10000 # l is the number of iterations
alpha = .01 # stepsize paramter
x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
r_t = np.array([]) # record of rewards
G = np.random.randint(-n,n,size=(n,n)) # a random grid
for i in range(l):
    r_t = np.append(r_t, x.transpose() @ G @ y)
    tmp = x.copy()
    xbar = np.zeros((n,1))
    xbar[np.argmax(G @ y)] = 1
    ybar = np.zeros((n,1))
    ybar[np.argmin(G.transpose() @ tmp)] = 1
    x = x + alpha*(xbar - x)
    y = y + alpha*(ybar - y)
    t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()