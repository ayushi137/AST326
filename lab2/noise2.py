import numpy as np
import matplotlib.pyplot as plt


######################### Functions ###########################################
# function to get mean and SD (can modify SD defination)
def mean_SD (data):
    # mean
    answer = np.zeros(2)
    mean = np.sum(data/np.size(data))
    answer[0] = mean
    # standard deviation
    hold = []
    for x in range(data.size):
        hold.append((data[x]-mean)**2)
    hold = np.array(hold)
    SD = (np.sqrt(sum(hold)/((hold.size) - 1)))
    variance = SD**2
    answer[1] = variance
    return answer

######################## load text###########################################

# check these value with other file noise.py
####
sample = np.arange(1,1001,1)
pick = [500, 750,900, 1000,1100,1250,1400,1500,1600, 1750, 1800]
n = len(pick)
intensity = []
M_V=[]
i = 0
while i <n:
    data = np.loadtxt("1000intensity_{0}pixel.txt".format(n), usecols=(i,))
    mean = mean_SD(data)
    intensity.append(data)
    M_V.append(mean)
    i+=1

mean = []
variance = []
i = 0
while i <n:
    mean.append(M_V[i][0])
    variance.append(M_V[i][1])
    i+=1

plt.figure(1)
plt.scatter(mean, variance)
plt.title("For selected Pixels")
plt.xlabel("mean")
plt.ylabel("variance")
plt.savefig("mean_V1.pdf")
plt.show()
