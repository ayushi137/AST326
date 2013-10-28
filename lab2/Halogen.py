import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("centroid_for_Vega1.txt", usecols=(0,))
y = np.loadtxt("centroid_for_Vega1.txt", usecols=(1,))


plt.scatter(x,y)
plt.show()