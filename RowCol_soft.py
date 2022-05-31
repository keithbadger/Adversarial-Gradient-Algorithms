import numpy as np
import matplotlib.pyplot as plt
n = 3 # n is the number of rows and columns
l = 100 # l is the number of iterations
alpha = .1 # stepsize paramter
x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
xP = np.array([])
yP = np.array([])
for i in range(n):
    xP = np.append(xP, np.exp(x[i]))
    yP = np.append(yP, np.exp(y[i]))
xP = xP.transpose() / xP.sum()
yP = yP.transpose() / yP.sum()
print(x)
print(xP)