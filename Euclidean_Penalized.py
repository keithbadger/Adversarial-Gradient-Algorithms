import numpy as np
import matplotlib.pyplot as plt

n = 3 # n is the number of rows and columns
l = 1000 # l is the number of iterations
alpha = .01 #stepsize parameter
e = .01
scale = 1
x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
x = np.array([[1],[0],[0]])
y = np.array([[0],[1],[0]])
r_t = np.array([]) # record of rewards
G = np.random.randint(-n,n,size=(n,n)) # a random grid
G = np.array([ [0,-1,1], [1,0,-1], [-1,1,0] ])

for k in range(l):
    alpha = scale/(scale + k)
    e = scale/(100*scale + k)
    r_t = np.append(r_t, x.transpose() @ G @ y)
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

#"""
disp = np.zeros((n,2))
for p in range(n):
    disp[p,0] = round(x[p][0],2)
    disp[p,1] = round(y[p][0],2)
print(disp)
#"""
t = np.array([list(range(l))])
plt.plot(t.transpose(),r_t)
plt.show()
