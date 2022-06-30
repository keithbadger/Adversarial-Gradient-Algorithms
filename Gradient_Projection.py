import numpy as np
import matplotlib.pyplot as plt
n = 3 # n is the number of rows and columns
l = 10000 # l is the number of iterations
alpha = .01 # stepsize paramter
x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
x = np.array([[1],[0],[0]])
y = np.array([[0],[1],[0]])
r_t = np.array([]) # record of rewards
x_t = np.array([])
y_t = np.array([])
z_t = np.array([])
G = np.array([ [0,-1,1], [1,0,-1], [-1,1,0] ])
#G = np.random.randint(-n,n,size=(n,n)) # a random grid
for i in range(l):
    r_t = np.append(r_t, x.transpose() @ G @ y)
    x_t = np.append(x_t, x[0])
    y_t = np.append(y_t, x[1])
    z_t = np.append(z_t, x[2])
    tmp = x.copy()
    xbar = x + G @ y
    xbar = xbar + (1-xbar.sum())/n
    arg = np.flip(np.argsort(xbar, axis=0))
    s = np.flip(np.sort(xbar, axis=0))
    dif = s[0]-1
    k = 1
    while k < n and s[k] > dif:
        k += 1
        dif += (s[k-1] - dif)/k
    while k < n:
        xbar[arg[k]] = dif
        k += 1
    xbar = xbar - dif
    x = x + alpha*(xbar - x)
    ybar = y - G.transpose() @ tmp
    ybar = ybar + (1-ybar.sum())/n
    arg = np.flip(np.argsort(ybar, axis=0))
    s = np.flip(np.sort(ybar, axis=0))
    dif = s[0]-1
    k = 1
    while k < n and s[k] > dif:
        k += 1
        dif += (s[k-1] - dif)/k
    while k < n:
        ybar[arg[k]] = dif
        k += 1
    ybar = ybar - dif
    y = y + alpha*(ybar - y)
print(x)
print(y)
t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()
plt.plot(t.transpose(),x_t)
plt.show()
plt.plot(t.transpose(),y_t)
plt.show()
plt.plot(t.transpose(),z_t)
plt.show()