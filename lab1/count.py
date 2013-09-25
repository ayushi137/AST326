from PMT import*
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import math

x = 1
data= [] 
'''
# Question 1 - 5
while x <= 6:
    data1 = np.loadtxt("test_{0}_0.001_100.dat".format(x))
    data.append(data1)
    x+=1
'''
# Question 8A
while x <= 6:
    data1 = np.loadtxt("SMALLRATEtask8pmtrunno_{0}count0.001.dat".format(x))
    data.append(data1)
    x+=1
'''
# Question 8B
while x <= 6:
    data1 = np.loadtxt("LARGESETpmtrunno_{0}.dat".format(x))
    data.append(data1)
    x+=1
'''

tsamp = 0.001
nsamp = 400


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

#### meanSD_rate is the mean and SD of the count rate by mean

# ~~~~~~~~~~~~~~~~~~~MOM ans SDOM
m3 = np.zeros(meanSD.shape[0])
for x in range(meanSD.shape[0]):
    m3[x] = meanSD[x][0]
MOM_SDOM = mean_SD(m3)

# ~~~~~~~~~~~~~~~~~~ theoretical distribution poission and gaussian
poission_the = []
gaussian_the = []
hmin = 0
hmax = 10
'''
for v in range(len(data)):
    hmin = min(data[v])
    hmax = max(data[v])
    print hmax
    w = np.arange(hmin, hmax+1, 0.001)
    P = ((meanSD[v][0])**w)*(np.exp(-(meanSD[v][0]))/sc.factorial(w))
    p = P*len(data[v])
    poission_the.append(p)
    
    G = (np.exp((-1/2)*(((w-(meanSD[v][0]))/(meanSD[v][1]))**2)))/((meanSD[v][1])*(np.sqrt(2*np.pi)))
    g = G*len(data[v])
    gaussian_the.append(g)
'''
w = np.arange(hmin, hmax+1, 0.001)
P2 = ((MOM_SDOM[0])**w)*(np.exp(-(MOM_SDOM[0]))/sc.factorial(w))
p2 = P2*len(data[0])
    
G2 = (np.exp((-1/2)*(((w-(MOM_SDOM[0]))/(MOM_SDOM[1]))**2)))/((MOM_SDOM[1])*(np.sqrt(2*np.pi)))
g2 = G2*len(data[0])



###### gprahs ... the setting changes ad per the data 
colors = "bgrcmykw"
c = 1
'''
# ~~~~~~~~~~~~~~~~~~~to plot the graph
plt.figure(c,figsize=(13, 8.5))
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
'''
# ~~~~~~~~~~~~~~~~to plot the histograms

plt.figure(c,figsize=(11, 9))
color_index = 0

hist_all = []
hmin = 0
hmax = 10
'''
for x in range(len(data)):
    #hmin = min(data[x])
    #hmax = max(data[x])
    print hmax
    hr = np.arange(hmin, hmax+1)
    o = np.arange(hmin, hmax+1, 0.001)
    #plt.figure(c,figsize=(11, 3))
    #plt.subplot(3,2,x+1)
    hist = np.array([np.where(data[x] ==i)[0].size for i in hr])
    hist_all.append(hist)
    plt.plot(hr,hist, drawstyle='steps-mid', lw=1.5 , c=colors[color_index]  )#label = 'Histogram')#
    color_index += 1
    plt.ylabel ('Frequency')
    #plt.plot(o,poission_the[x], label = 'Poisson')
    #plt.plot(o,gaussian_the[x], label= 'Gaussian')
'''
o = np.arange(hmin, hmax+1, 0.001)
plt.plot(o, p2, lw=2, label = 'Poisson')
plt.plot(o, G2*100, lw=2, label = 'Gaussian')
plt.legend()
plt.xlabel ('Count')

#plt.savefig('Histogram_with_distribution_0.01_400.pdf')
    #c+=1
'''
#~~~~~~~~~~~~~~plot mean and standard deviation
c+=1
plt.figure(c,figsize=(9, 6))
xbar = []
s = []
for i in range(len(data)):
    xbar.append (meanSD[i][0])
    s.append(meanSD[i][1])
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
'''

plt.show()


