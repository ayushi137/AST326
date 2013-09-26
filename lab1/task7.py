import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import math


tspam = [0.01, 0.0125, 0.025, 0.0375, 0.05]
data=[]
for y in tspam:
    print y
    x =1
    data1 = []
    while x <= 6:
        dat = np.loadtxt("LONGCOUNTtask7pmtrunno_{0}count{1}.dat".format(x,y))
        data1.append(dat)
        x+=1
    data.append(data1)

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
    # which one of these two are better????????
    SD = (math.sqrt(sum(hold)/((hold.size) - 1)))
    # SD = (math.sqrt(sum(hold)/((hold.size) - 1)))/math.sqrt(hold.size)
    answer[1] = SD
    return answer

MSD = np.zeros((5,6,2))
w=0
for y in data:
    meanSD = np.zeros((len(y),2))
    for x in range(len(y)):    
        meanSD[x] = mean_SD(y[x])
    MSD[w]=meanSD
    w+=1

plt.figure(1,figsize=(9, 6))
colors = "bgrcm"
color_index = 0
for a in MSD:
    xbar = []
    s = []
    for i in range(6):
        xbar.append (a[i][0])
        s.append(a[i][1])
    xbar = np.array(xbar)
    s = np.array(s)
    s2 = s**2 
    plt.scatter(xbar, s2, c=colors[color_index], label = "Time: {0}s".format(tspam[color_index]) )
    color_index += 1

x = np.arange(0 , 100, 0.1);
y = x
plt.plot(x, y, 'k', label = "x=y graph")
plt.xlim(0, 100)
plt.ylim(0,100)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel("Mean")
#plt.ylabel("Variance (Standard deviation squared)")
plt.legend(loc =2)
plt.savefig("graphs/Task7.pdf")
plt.show()
