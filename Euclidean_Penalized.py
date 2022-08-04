import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

index = 1  # index of given grid size
n = 8 # n is the number of rows and columns
l = 1000 # l is the number of iterations
alpha = .01 #stepsize parameter
e = .5
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
    grad = -1*(G @ y)
    args = np.flip(np.argsort((grad - 2*e*x).transpose()[0]),axis=0)
    for j in range(n):
        xbar = np.zeros((n,1))  
        xbar[args[j:]] = (2*e - (2*e*x[args[j:]]-grad[args[j:]]).sum() + (n-j)*(2*e*x[args[j:]] - grad[args[j:]]))/(2*e*(n-j))
        if xbar.min() >= 0:
            break
    grad = G.transpose()@x
    args = np.flip(np.argsort((grad - 2*e*y).transpose()[0]),axis=0)
    for i in range(n):
        ybar = np.zeros((n,1))  
        ybar[args[i:]] = (2*e - (2*e*y[args[i:]]-grad[args[i:]]).sum() + (n-i)*(2*e*y[args[i:]] - grad[args[i:]]))/(2*e*(n-i))
        if ybar.min() >= 0:
            break
    """
    print()
    print("------",k+1,"--------")
    print()
    disp = np.zeros((n,2))
    for p in range(n):
        disp[p,0] = round(x[p][0],2)
        disp[p,1] = round(y[p][0],2)
    print(disp)
    print() 
    """ 
    x = x + alpha*(xbar - x)
    y = y + alpha*(ybar - y)

upper = opt.linprog(-G@y , A_eq=np.ones((1,n)), b_eq=[1])
lower = opt.linprog(G.T@x, A_eq=np.ones((1,n)), b_eq=[1])

#"""
print("+===============+")
print("|   EUCLIDEAN   |")
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
