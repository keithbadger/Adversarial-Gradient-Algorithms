import numpy as np
import matplotlib.pyplot as plt

index = 3 # index of given grid size
n = 8 # n is the number of rows and columns
l = 1000 # l is the number of iterations
alpha = .01 #stepsize parameter
scale = 5

x = np.random.rand(n,1) # is the probability of selecting a row with maximization goal
y = np.random.rand(n,1) # is the probability of selecting a col with minimization goal
x = x / x.sum()
y = y / y.sum()
x_sum = np.zeros((n,1))
y_sum = np.zeros((n,1))
r_t = np.array([]) # record of rewards
u_t = np.array([]) # record of upperbounds
l_t = np.array([]) # record of lowerbounds
"""-----------------
x = np.array([[.999],[0.0005],[0.0005]])
y = np.array([[0.0005],[.999],[0.0005]])
x0 = np.array([])
x1 = np.array([])
x2 = np.array([])
y0 = np.array([])
y1 = np.array([])
y2 = np.array([])
#"""
game_file = open(str(n)+'x'+str(n)+ " #" + str(index), 'r')
#game_file = open("RPS")
G = np.loadtxt(game_file)
game_file.close()

for k in range(l):
    #alpha = scale/(scale+i)
    tmp = x.copy()
    xbar = np.zeros((n,1))
    xbar[np.argmax(G @ y)] = 1
    ybar = np.zeros((n,1))
    ybar[np.argmin(G.transpose() @ tmp)] = 1
    x = x + alpha*(xbar - x)
    y = y + alpha*(ybar - y)
    x_sum = x_sum + x
    y_sum = y_sum + y
    xa = x_sum/(k+1)
    ya = y_sum/(k+1)
    r_t = np.append(r_t, xa.transpose() @ G @ ya)
    u_t = np.append(u_t,np.max(G@ya))
    l_t = np.append(l_t,np.min(G.T@xa))
    """-----------------
    x0 = np.append(x0, x[0])
    x1 = np.append(x1, x[1])
    x2 = np.append(x2, x[2])
    y0 = np.append(y0, y[0])
    y1 = np.append(y1, y[1])
    y2 = np.append(y2, y[2])
    #"""
upper = np.max(G@ya)
lower = np.min(G.T@xa)

#"""-------------------
print("+===============+")
print("|  CONDITIONAL  |")
print("+===============+")
print("| ROW   |   COL |")
print("+===============+")
for p in range(n):
    print(f"| {xa[p][0]:.2f}  |  {ya[p][0]:.2f} |")
print("+===============+")
print(f"|REWARD: {(xa.T@G@ya)[0][0]:7.2f}|")
print(f"|UPPER:  {upper:7.2f}|")
print(f"|LOWER:  {lower:7.2f}|")
print("+===============+")
print(f"|GAP:    {upper - lower:7.2f}|")
print("+===============+")
#"""
#"""------------------
t = np.array([list(range(l))])
plt.plot(t.T,r_t,'b',t.T,u_t,'g',t.T,l_t,'r')
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.legend(["Reward","Upperbound","Lowerbound"])
plt.show()
#"""
"""------------------
ax = plt.axes(projection='3d')
ax.plot3D(np.array([1,0,0,1]),np.array([0,1,0,0]),np.array([0,0,1,0]),'black')
ax.plot3D(x0,x1,x2,'green')
ax.plot3D(y0,y1,y2,'red')
ax.set_xlabel("rock")
ax.set_ylabel("paper")
ax.set_zlabel("scissors")
plt.show()
#"""