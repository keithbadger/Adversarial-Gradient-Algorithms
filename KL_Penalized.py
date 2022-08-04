import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

index = 1  # index of given grid size
n = 8 # n is the number of rows and columns
l = 1000 # l is the number of iterations
alpha = .01 #stepsize parameter
e = 1 # penalty coeff
scale = 5

x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
r_t = np.array([]) # record of rewards
u_t = np.array([]) # record of upperbounds
l_t = np.array([]) # record of lowerbounds
game_file = open(str(n)+'x'+str(n)+ " #" + str(index), 'r')
G = np.loadtxt(game_file)
game_file.close()

for k in range(l):
    alpha = scale/(scale + k)
    r_t = np.append(r_t, x.transpose() @ G @ y)
    u_t = np.append(u_t,((opt.linprog(-G@y , A_eq=np.ones((1,n)), b_eq=[1])).x@G@y)[0])
    l_t = np.append(l_t,x.T@G@np.array([(opt.linprog(G.T@x, A_eq=np.ones((1,n)), b_eq=[1])).x]).T)
    xbar = x * np.exp(G @ y / e)
    xbar = xbar/xbar.sum()
    ybar = y * np.exp(-G.T @ x / e)
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
plt.plot(t.T,r_t,'b',t.T,u_t,'g',t.T,l_t,'r')
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.legend(["Reward","Upperbound","Lowerbound"])
plt.show()
