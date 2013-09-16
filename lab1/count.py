from PMT import*
import numpy as np
import matplotlib.pyplot as plt

nsamp = 0.1
tsamp = 10

x = np.loadtxt("data3.txt")

#count = 0

#for i in x:
#    count = count + i/tsamp
    
#print count/nsamp

# to plot the graph
plt.figure()
plt.plot (x)
plt.show()


