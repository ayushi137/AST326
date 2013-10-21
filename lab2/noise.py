# To read noice and get the errors 

import numpy as np
import matplotlib.pyplot as plt

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
    
    pixel = np.loadtxt("1000_Lamp/100ms_Lamp_{0}{1}.txt".format(k,n),usecols=(0,)) #, skiprows = 16, comments = ">")
    pixellist.append(pixel)
    intensity = np.loadtxt("1000_Lamp/100ms_Lamp_{0}{1}.txt".format(k,n),usecols=(1,))#, skiprows = 16, comments = ">")
    intensitylist.append(intensity)


######################### specific pixel ######################################
sample = trial
i = 0
# as the list change change in other file as well
pick = [500, 750,900, 1000,1100,1250,1400,1500,1600, 1750, 1800]
inten = np.zeros([len(pick),1000])
j = 0
while j<len(pick):
    i=0
    while i< 1000:
        inten[j][i] = intensitylist[i][pick[j]]
        i+=1
    j+=1

output = np.transpose(inten)

np.savetxt("1000intensity_{}pixel.txt".format(len(pick)),output,fmt='%0.2f')

plt.figure(1, figsize=(8,9))
plt.subplot(5,1,1)
plt.plot(sample, inten[0], label = "pixel = {0}".format(pick[0]))
plt.legend(prop={'size':10})
plt.subplot(5,1,2)
plt.plot(sample, inten[1], label = "pixel = {0}".format(pick[1]))
plt.legend(prop={'size':10})
plt.subplot(5,1,3)
plt.plot(sample, inten[2], label = "pixel = {0}".format(pick[2]))
plt.legend(prop={'size':10})
plt.subplot(5,1,4)
plt.plot(sample, inten[3], label = "pixel = {0}".format(pick[3]))
plt.legend(prop={'size':10})
plt.subplot(5,1,5)
plt.plot(sample, inten[4], label = "pixel = {0}".format(pick[4]))
plt.xlabel("sample")
plt.legend(prop={'size':10})
plt.tight_layout()
#plt.savefig("Noise3-1.pdf")

plt.figure(2, figsize=(8,9))
plt.subplot(6,1,1)
plt.plot(sample, inten[5], label = "pixel = {0}".format(pick[5]))
plt.legend(prop={'size':10})
plt.subplot(6,1,2)
plt.plot(sample, inten[6], label = "pixel = {0}".format(pick[6]))
plt.legend(prop={'size':10})
plt.subplot(6,1,3)
plt.plot(sample, inten[7], label = "pixel = {0}".format(pick[7]))
plt.legend(prop={'size':10})
plt.subplot(6,1,4)
plt.plot(sample, inten[8], label = "pixel = {0}".format(pick[8]))
plt.legend(prop={'size':10})
plt.subplot(6,1,5)
plt.plot(sample, inten[9], label = "pixel = {0}".format(pick[9]))
plt.legend(prop={'size':10})
plt.subplot(6,1,6)
plt.plot(sample, inten[10], label = "pixel = {0}".format(pick[10]))
plt.xlabel("sample")
plt.legend(prop={'size':10})
plt.tight_layout()
#plt.savefig("Noise3-2.pdf")

plt.figure(3)
plt.plot(pixellist[76], intensitylist[76])
plt.xlim(500,900)
plt.savefig("plot3_1.pdf")

plt.show()