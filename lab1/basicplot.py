from PMT import*
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import math

x = 1
data= [] 

tsamp=0.01
'''
# Question 1 - 5
text = ["test_1_0.001_100.dat","SMALLRATEtask8pmtrunno_1count0.001.dat","LARGESETpmtrunno_1.dat","LONGCOUNTtask7pmtrunno_1count0.05.dat"]
for x in text:
    data1 = np.loadtxt(x)
    data.append(data1)
'''
text =[[0.001,100],[0.001,1000],[0.01,400]]
text = [1 , 2, 3, 4, 5, 6]
for x in text:
    data1 = np.loadtxt("test_{0}_0.001_100.dat".format(x))
    data.append(data1)



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
count_rate = meanSD/tsamp             # count rate is x/time
m2 = np.zeros(count_rate.shape[0])
for x in range(count_rate.shape[0]):
    m2[x] = count_rate[x][0]
    
meanSD_rate = mean_SD(m2)


#######################################################
colors = "bgrmcykw"
c_i = 0
c = 1
# ~~~~~~~~~~~~~~~~~~~to plot the graph
labels = ["0.001s", "0.001s", "0.01s", "0.05s"]
plt.figure(c,figsize=(13, 8))
for x in range(len(data)):
    #plt.figure(c,figsize=(13, 2))
    plt.subplot(len(data),1,x+1)
    plt.plot (data[x], drawstyle = 'steps-mid', color='g')#color=colors[c_i], label =labels[c_i])
    plt.ylim (0,max(data[x]),2)
    '''
    if c_i == 3:
        plt.ylim(20, 140)
    plt.legend()
    c_i += 1
    #c+=1
    '''
#plt.ylim (0,6)
    plt.ylabel('Count')
plt.xlabel('Time(ms)')
plt.tight_layout()

plt.savefig ("graphs/task1.pdf")
c+=1
'''

#~~~~~~~~~~~~~~~~
c_i = 0
plt.figure(c,figsize=(13, 5))

hist_all = []
hmin = 0
hmax = 35
labels = ["0.001 for 100", "0.001 for 1000", "0.01 for 400", "0.05 for 400"]
for x in range(len(data)):
    hmin = min(data[x])
    hmax = max(data[x])
    hr = np.arange(hmin, hmax+1)
    o = np.arange(hmin, hmax+1, 0.001)
    #plt.figure(c,figsize=(11, 3))
    plt.subplot(len(data),1,x+1)
    hist = np.array([np.where(data[x] ==i)[0].size for i in hr])
    hist_all.append(hist)
    plt.plot(hr,hist, drawstyle='steps-mid', lw=1.5, color=colors[c_i], label =labels[c_i]  )
    plt.ylabel ('Frequency')
    #plt.ylim(0,50)
    plt.legend()
    c_i += 1
plt.xlabel ('Count')
plt.tight_layout()
#plt.savefig ("graphs/dark_hist.pdf")
'''

plt.show()
