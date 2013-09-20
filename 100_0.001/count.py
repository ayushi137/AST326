from PMT import*
import numpy as np
import matplotlib.pyplot as plt
import math

data1 = np.loadtxt("test_1_0.001_100.dat")
data2 = np.loadtxt("test_1_0.001_100.dat")
data3 = np.loadtxt("test_1_0.001_100.dat")
data4 = np.loadtxt("test_1_0.001_100.dat")
data5 = np.loadtxt("test_1_0.001_100.dat")
data6 = np.loadtxt("test_1_0.001_100.dat")
data = [data1,data2,data3,data4,data5,data6]

tsamp = 0.001
nsamp = 100


# ~~~~~~~~~~~~~~~~~~~~ mean and standard deviation

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

m1 = np.zeros((6,2))
for x in range(len(data)):    
    m1[x] = mean_SD(data[x])

# ~~~~~~~~~~~~~~~~~~~~~ mean count rate
count_rate = m1/tsamp 
m2 = np.zeros(count_rate.shape[0])
for x in range(count_rate.shape[0]):
    m2[x] = count_rate[x][0]
    
meanSD_rate = mean_SD (m2)



# ~~~~~~~~~~~~~~~~~~~to plot the graph
plt.figure(1,figsize=(9, 6))
for x in range(len(data)):

    plt.subplot(3,2,x+1)
    plt.plot (data[x], 'm', drawstyle = 'steps-mid')
    plt.xlabel('Count')
    plt.ylabel('Time (ms)')


# ~~~~~~~~~~~~~~~~to plot the histograms
hmin = 0
hmax = 5

plt.figure(2)
hr = np.arange(hmin, hmax+1)

hist_all = []

for x in range(len(data)):
    plt.subplot(3,2,x+1)
    hist = np.array([np.where(data[x] ==i)[0].size for i in hr])
    hist_all.append(hist)
    plt.plot(hr,hist, drawstyle='steps-mid')

#~~~~~~~~~~~~~~plot mean and standard deviation
plt.figure(3)
xbar = []
s = []
for i in range(len(data)):
    xbar.append (m1[i][0])
    s.append(m1[i][1])
xbar = np.array(xbar)
s = np.array(s)
s2 = s**2 
plt.plot(xbar, s2, 'ro' )

x = np.arange(0.5, 1.5, 0.1);
y = x
plt.plot(x, y)
plt.yscale('log')
plt.xscale('log')

#~~~~~~~~~~~poission distribution


plt.show()


