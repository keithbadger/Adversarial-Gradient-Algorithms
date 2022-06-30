import numpy as np
import matplotlib.pyplot as plt
n = 3 # n is the number of rows and columns
l = 10 # l is the number of iterations
alpha = 1 # stepsize paramter
x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = np.array([[4],[0],[0]])
y = np.array([[0],[4],[0]])
r_t = np.array([]) # record of rewards
G = np.random.randint(-n,n,size=(n,n)) # a random grid
G = np.array([ [0,-1,1], [1,0,-1], [-1,1,0] ])
for i in range(l):
    xP = np.exp(x)
    yP = np.exp(y)
    xP = xP / xP.sum()
    yP = yP / yP.sum()
    x = x + alpha*((np.identity(n)*xP - xP@xP.transpose()) @ G @ yP)
    y = y - alpha*((np.identity(n)*yP - yP@yP.transpose()) @ G.transpose() @ xP)
    r_t = np.append(r_t, xP.transpose() @ G @ yP)
    print(xP)
    print(yP)
    print("///////////////")
t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()

