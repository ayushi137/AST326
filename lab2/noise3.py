# To read noice and get the errors 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

######################### loading files ######################################

pixellist = []
intensitylist = []
trial = np.arange(1,1001,1) ### change in other file

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
    
    pixel = np.loadtxt("1000_Lamp/100ms_Lamp_{0}{1}.txt".format(k,n),usecols=(0,))
    pixellist.append(pixel)
    intensity = np.loadtxt("1000_Lamp/100ms_Lamp_{0}{1}.txt".format(k,n),usecols=(1,))
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
pick = np.arange(500,900,10)
n = len(pick)
inten = np.zeros([n,1000])
M_V = []
j = 0
while j<n:
    i=0
    while i< 1000:
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

mean = np.array(mean)
variance = np.array(variance)
############################## fit ###########################################
p0 = [-3380,1.44]
def peval(mean, p):
    return (p[0]+p[1]*mean)

def residuals (p,variance, mean, peval):
    return (variance) - peval(mean,p)

p_final = leastsq(residuals,p0,args=(variance,mean, peval), full_output= True,maxfev=2000)


if p_final[4] == 1: # change p_final[1] to success
    print "It converges."
else:
    print "It does not converge."

output = np.column_stack((mean,variance))
#np.savetxt("meanVariance.csv", output, delimiter=',', fmt='%.2f')

############################### plot ########################################
plt.figure(1)
plt.scatter(mean, variance)
plt.plot(mean, peval(mean, p_final[0]),color='r')

plt.text(8000, 60000, 'k = 1.44')

plt.text(8000, 57000, 's_o^2 = -3380')
plt.title("Pixel from 500 to 900")
plt.xlabel("mean")
plt.ylabel("variance")
plt.savefig("mean_Variance3_reduced.pdf")
plt.show()
