import numpy as np
import matplotlib.pyplot as plt
p1 = np.array([[1], [0], [0]]) #these are the initial probabilities of rock paper and scissors respectively for player 1
p2 = np.array([[0], [1], [0]]) # same thing for player 2
y = np.array([])
l=100 # the number of iterations the policy gradient runs
for i in range(l):
    print("Player1: ", p1.transpose(), "\nPlayer2: ", p2.transpose())
    A = np.array([ [0,-1,1], [1,0,-1], [-1,1,0] ]) #this is the reward matrix essentially
    r = (p1.transpose() @ A @ p2)[0][0] #and the reward is computed here
    print("Expected Player 1 Reward: ", r , "\nExpected Player 2 Reward: ", -r) 
    alpha = .1  # the stepsize is defined here
    tmp = p1.copy()
    p1 = p1 + alpha * A @ p2 # player 1 update
    p2 = p2 - alpha * A.transpose() @ tmp # player 2 update
    for x in range(3):
        if p1[x] > 1:
            p1[x] = 1
        if p1[x] < 0:
            p1[x] = 0
        if p2[x] > 1:
            p2[x] = 1
        if p2[x] < 0:
            p2[x] = 0 # this just trims all the values between 0 and 1. There is probably a better way of doing this :(
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum() 
    y = np.append(y,r) # the values are being appended to y so that they can be graphed later
x = np.array([list(range(l))])
plt.plot(x.transpose(),y)
plt.show()
