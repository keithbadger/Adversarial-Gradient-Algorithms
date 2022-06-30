import numpy as np


for i in range(21):
    n = 2 + i//3
    G = np.random.randint(-n,n,size=(n,n))
    filename = str(n) + "x" + str(n) + " #" + str(i%3+1)
    file = open(filename, "w+")
    content = str(G)
    file.write(content)
    file.close()