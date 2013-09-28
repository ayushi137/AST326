from PMT import*
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import math

x = 1
data= [] 

tsamp=100

# Question 1 - 5
while x <= 6:
    data1 = np.loadtxt("dark_{0}_0.001_{1}.dat".format(x, tsamp))
    data.append(data1)
    x+=1

# ~~~~~~~~~~~~~~~~~~~~ mean and standard deviation 

# this can be used for any data

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

meanSD = np.zeros((len(data),2))
for x in range(len(data)):    
    meanSD[x] = mean_SD(data[x])

#### meanSD is an array which has mean and standard deviation of each data set

# ~~~~~~~~~~~~~~~~~~~~~ mean count rate

count_rate = meanSD/tsamp              # count rate is x/time
m2 = np.zeros(count_rate.shape[0])
for x in range(count_rate.shape[0]):
    m2[x] = count_rate[x][0]
    
meanSD_rate = mean_SD(m2)


#######################################################
colors = "bgrcmykw"
c = 1
# ~~~~~~~~~~~~~~~~~~~to plot the graph
plt.figure(c,figsize=(9, 8))
for x in range(len(data)):
    #plt.figure(c,figsize=(11, 3))
    plt.subplot(6,1,x+1)
    plt.plot (data[x], 'm', drawstyle = 'steps-mid')
    plt.ylim (0,max(data[x]),2)
    plt.ylabel('Time (ms)')
    plt.tight_layout()
    c+=1
plt.xlabel('Count')
c+=1


plt.figure(c,figsize=(13, 9))

hist_all = []
hmin = 0
hmax = 35

for x in range(len(data)):
    hmin = min(data[x])
    hmax = max(data[x])
    hr = np.arange(hmin, hmax+1)
    o = np.arange(hmin, hmax+1, 0.001)
    #plt.figure(c,figsize=(11, 3))
    plt.subplot(3,2,x+1)
    hist = np.array([np.where(data[x] ==i)[0].size for i in hr])
    hist_all.append(hist)
    plt.plot(hr,hist, drawstyle='steps-mid', lw=1.5  )
    plt.ylabel ('Frequency')


plt.show()
