import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import math

nsamp = [2,4,8,16,32,64,128,256,512,1024,2048]

data=[]
for y in nsamp:
    print y
    x =1
    data1 = []
    while x <= 10:
        dat = np.loadtxt("TASK9pmtrunno_{0}sample{1}.dat".format(x,y))
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

MSD = np.zeros((11,10,2))
w=0
for y in data:
    meanSD = np.zeros((len(y),2))
    for x in range(len(y)):    
        meanSD[x] = mean_SD(y[x])
    MSD[w]=meanSD
    w+=1

MSDOM = np.zeros((11,2))
h=0
for e in MSD:
    m1 = np.zeros(10)
    for x in range(10):
        m1[x] = e[x][0]
    MSDOM[h] = mean_SD(m1)
    h+=1

mean = []
SD = []
SD_predict = []
t = 0
for q in MSDOM:
    SD_predict.append(q[0]/(np.sqrt(nsamp[t])))
    mean.append(q[0])
    SD.append(q[1])
    t+=1
mean = np.array(mean)

SD_predict = np.sqrt(mean)

plt.figure(1,figsize=(9, 6))
plt.plot(nsamp,mean,lw=2, c='r', label = 'Mean of mean')
plt.plot(nsamp,SD, lw=2, c='g' , label = 'Standard Deviation of mean')
plt.plot(nsamp,SD_predict,  lw=2 , c='k' , label = 'Theoretical Standard Deviation')
plt.ylim(-0.5, 3)
plt.xlim(0,2050)
plt.legend()

plt.show()
