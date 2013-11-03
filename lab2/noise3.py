# To read noice and get the errors 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

d = []
trial2 = np.arange(1,11,1)
for n in trial2:
    k = ""
    if n < 10:
        k = "0000"
    elif n >= 10 and n < 100:
        k = "000"
    elif n >= 100 and n <1000:
        k = "00"
    else:
        k = "0"
    dark = np.loadtxt("Dark/100ms_Dark{0}{1}.txt".format(k,n),usecols=(1,))
    d.append(dark)

d = np.array(d)
d = np.transpose(d)
dark = []
for i in range(len(d)):
    dark.append(np.average(d[i]))
dark = np.array(dark)

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
pick = np.arange(0,2048,1)
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
index = []
for i in range(len(pick)):
    mean[i] = mean[i] -dark[pick[i]]
        #if mean[i] > 40000:
#index.append(i)

#mean = [i for j, i in enumerate(mean) if j not in index]
#variance = [i for j, i in enumerate(variance) if j not in index]

############################## fit ###########################################
p0 = [-1883,1.42]
def peval(mean, p):
    return (p[0]+p[1]*mean)

def residuals (p,variance, mean, peval):
    return (variance) - peval(mean,p)

p_final = leastsq(residuals,p0,args=(variance,mean, peval), full_output= True,maxfev=2000)

y_final = peval(mean,p_final[0])
chi2 = np.sum((variance - y_final)**2)#/ ((dy)**2))
resi = (residuals(p_final[0],variance,mean,peval))
dof = len(variance)-len(p0)
chi_re2 = chi2/dof # residual variance
#cov = p_final[1] * chi_re2
#cov_xy = cov[0][1]
#cov_x = np.sqrt(cov[0][0])
#cov_y = np.sqrt(cov[1][1])
#r =cov_xy/(cov_x*cov_y)

print "The inital parameter (p0) we used is:\n", p0
print "What we get as a parameter:", p_final[0]

if p_final[4] == 1: # change p_final[1] to success
    print "It converges."
else:
    print "It does not converge."

print "The Chi square is: \t\t\t",round(chi2,2)
print "The Chi-reduced square is: \t\t", round(chi_re2,2)
print
#print "Cov_xy:",round(cov_xy,4), "\nCov_x: ",round(cov_x,4),"\nCov_y: ", round(cov_y,4)
#print "Sample coefficient of linear correlation (r): ", round(r,2)
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print


output = np.column_stack((mean,variance))
#np.savetxt("meanVariance.csv", output, delimiter=',', fmt='%.2f')

############################### plot ########################################
plt.figure(1)
x = np.arange(0,len(mean),1)
plt.plot(x,mean)
#plt.plot(mean, peval(mean, p_final[0]),color='r')

#plt.text(5000, 55000, 'k = 1.37')

#plt.text(500, 4500, 's_o^2 = 1027')
plt.title("Spectrum of Lamp")
plt.xlabel("Pixel")
plt.xlim(0,2060)
plt.ylim(0)
plt.ylabel("Intensity")
#plt.savefig("mean_Variance3_reduced.pdf")

#plt.figure(2)
#plt.plot(mean)

plt.show()
