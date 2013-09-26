from PMT import*
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import math

x = 1
data= [] 

# Question 8
while x <= 6:
    data1 = np.loadtxt("SMALLRATEtask8pmtrunno_{0}count0.001.dat".format(x))
    data.append(data1)
    x+=1

#LARGESETpmtrunno_{0}
#SMALLRATEtask8pmtrunno_{0}count0.001

tsamp = 0.001
nsamp = 100


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
m3 = np.zeros(meanSD.shape[0])
for x in range(meanSD.shape[0]):
    m3[x] = meanSD[x][0]
MOM_SDOM = mean_SD(m3)
# ~~~~~~~~~~~~~~~~~~ theoretical distribution poission and gaussian
'''
hmin = 0
hmax = 30
w = np.arange(hmin, hmax+1, 0.001)
G = (np.exp((-1/2)*(((w-(MOM_SDOM[0]))/(4.13))**2)))/((4.13)*(np.sqrt(2*np.pi)))
g = G*len(data[0])

P = ((MOM_SDOM[0])**w)*(np.exp(-(MOM_SDOM[0]))/sc.factorial(w))
p = P*len(data[0])
 
'''
poission_the = []
gaussian_the = []
for v in range(len(data)):
    #v = 0
    hmin = 0
    hmax = max(data[v])
    w = np.arange(hmin, hmax+1, 0.001)

    P = ((meanSD[v][0])**w)*(np.exp(-(meanSD[v][0]))/sc.factorial(w))
    p = P*len(data[v])
    poission_the.append(p)
    
    G = (np.exp((-1/2)*(((w-(meanSD[v][0]))/(meanSD[v][1]))**2)))/((meanSD[v][1])*(np.sqrt(2*np.pi)))
    g = G*len(data[v])
    gaussian_the.append(g)


###### gprahs ... the setting changes ad per the data 
c = 1

# ~~~~~~~~~~~~~~~~to plot the histograms

plt.figure(c,figsize=(11, 9))

hist_all = []

#for x in range(len(data)):
    #x = 0
hmin = 0
hmax = max(data[x])
hr = np.arange(hmin, hmax+1)
w = np.arange(hmin, hmax+1, 0.001)
plt.figure(c,figsize=(11, 3))
    #plt.subplot(3,2,x+1)
hist = np.array([np.where(data[x] ==i)[0].size for i in hr])
hist_all.append(hist)
plt.plot(hr,hist, drawstyle='steps-mid', lw=2, color='k', label='Histogram' )
plt.ylabel ('Frequency')
plt.xlim (0,10)
plt.ylim (0,350)
plt.plot(w,poission_the[x], color='g', lw=2,label='Poisson')
plt.plot(w,gaussian_the[x], color= 'b', lw=2,label='Gaussian')
plt.vlines(meanSD[x][0],0,350, color= 'r', lw=2,label='Mean')
    #plt.vlines(meanSD[x][0] - meanSD[x][1],0,350, color= 'm', lw=1,label='Standard Deviation')
    #plt.vlines(meanSD[x][0] + meanSD[x][1],0,350, color= 'm', lw=1)
#plt.title ("1000 samples for 0.001s time spam")
plt.xlabel ('Count')
plt.legend(prop = {'size':15})
#   c+=1
#plt.tight_layout()
plt.savefig('Histogram_with_distribution_0.001_1000_3_.pdf')
'''
hmin = 0
hmax = 30
w = np.arange(hmin, hmax+1, 0.001)
plt.plot(w,p)
plt.plot(w,g)
plt.xlabel ('Count')
'''

plt.show()


