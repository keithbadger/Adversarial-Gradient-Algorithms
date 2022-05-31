import numpy as np
import matplotlib.pyplot as plt
n = 20 # n is the number of rows and columns
l = 100 # l is the number of iterations
alpha = .1 # stepsize paramter
x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
r_t = np.array([]) # record of rewards
G = np.random.randint(-n,n,size=(n,n)) # a random grid
for i in range(l):
    r_t = np.append(r_t, x.transpose() @ G @ y)
    tmp = x.copy()
    x = x + alpha * G @ y
    y = y - alpha * G.transpose() @ tmp
    x_min = x.min()
    y_min = y.min()
    if x_min < 0:
        x = x - x_min
    if y_min < 0:
        y = y - y_min
    x = x / x.sum()
    y = y / y.sum()
t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()

