import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

index = 1  # index of given grid size
n = 8 # n is the number of rows and columns
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

upper = opt.linprog(-G@y , A_eq=np.ones((1,n)), b_eq=[1])
lower = opt.linprog(G.T@x, A_eq=np.ones((1,n)), b_eq=[1])

#"""
print("+===============+")
print("|    SOFTMAX    |")
print("+===============+")
print("| ROW   |   COL |")
print("+===============+")
for p in range(n):
    print(f"| {x[p][0]:.2f}  |  {y[p][0]:.2f} |")
print("+===============+")
print(f"|REWARD: {(x.T@G@y)[0][0]:7.2f}|")
print(f"|UPPER:  {([upper.x]@G@y)[0][0]:7.2f}|")
print(f"|LOWER:  {(x.T@G@np.array([lower.x]).T)[0][0]:7.2f}|")
print("+===============+")
print(f"|GAP:    {([upper.x]@G@y)[0][0] - (x.T@G@np.array([lower.x]).T)[0][0]:7.2f}|")
print("+===============+")
#"""



t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()
