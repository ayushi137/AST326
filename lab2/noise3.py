# To read noice and get the errors 

import numpy as np
import matplotlib.pyplot as plt

######################### loading files ######################################

pixellist = []
intensitylist = []
trial = np.arange(1,101,1) ### change in other file

for n in trial:
    k = ""
    if n < 10:
        k = "0000"
    elif n >= 10 and n < 100:
        k = "000"
    elif n >= 100 and n <1000:
        k = "00"
    else:
        k = "0"
    
    pixel = np.loadtxt("100_lamp_data/100ms_Lamp_{0}{1}.txt".format(k,n),usecols=(0,), skiprows = 16, comments = ">")
    pixellist.append(pixel)
    intensity = np.loadtxt("100_lamp_data/100ms_Lamp_{0}{1}.txt".format(k,n),usecols=(1,), skiprows = 16, comments = ">")
    intensitylist.append(intensity)

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

######################### specific pixel ######################################
sample = trial
# as the list change change in other file as well
pick = np.arange(300,2000,1)
n = len(pick)
inten = np.zeros([n,100])
M_V = []
j = 0
while j<n:
    i=0
    while i< 100:
        inten[j][i] = intensitylist[i][pick[j]]
        i+=1
    mean = mean_SD(inten[j])
    M_V.append(mean)
    j+=1


mean = []
variance = []
m= 0
while m<n:
    mean.append(M_V[m][0])
    variance.append(M_V[m][1])
    m+=1

plt.figure(1)
plt.scatter(mean, variance)
plt.title("Pixel from 300 to 2000")
plt.xlabel("mean")
plt.ylabel("variance")
#plt.savefig("mean_Variance2.pdf")
plt.show()
