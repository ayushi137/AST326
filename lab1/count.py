from PMT import*
import numpy as np
import matplotlib.pyplot as plt
import math

data1 = np.loadtxt("lab1_data1.txt")
data2 = np.loadtxt("lab1_data2.txt")
data3 = np.loadtxt("lab1_data3.txt")
data4 = np.loadtxt("lab1_data4.txt")
data5 = np.loadtxt("lab1_data5.txt")
data6 = np.loadtxt("lab1_data6.txt")
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

'''
# ~~~~~~~~~~~~~~~~~~~to plot the graph
for x in range(len(data)):

    plt.figure(x+1, figsize=(9, 6))
    plt.plot (data[x], 'm')
    plt.xlabel('Count')
    plt.ylabel('Time (ms)')


# ~~~~~~~~~~~~~~~~to plot the histograms
hmin = 0
hmax = 5

hr = np.arange(hmin, hmax+1)
hist1 = np.array([np.where(data1 ==i)[0].size for i in hr])
hist2 = np.array([np.where(data2 ==i)[0].size for i in hr])
hist3 = np.array([np.where(data3 ==i)[0].size for i in hr])
hist4 = np.array([np.where(data4 ==i)[0].size for i in hr])
hist5 = np.array([np.where(data5 ==i)[0].size for i in hr])
hist6 = np.array([np.where(data6 ==i)[0].size for i in hr])
hist = [hist1,hist2,hist3,hist4,hist5,hist6]

for x in range(len(data)):
    plt.figure(7+x)
    plt.plot(hr,hist[x], drawstyle='steps-mid')
'''
#~~~~~~~~~~~~~~plot mean and standard deviation
plt.figure(1)
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

plt.show()


