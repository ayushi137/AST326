import numpy as np
import matplotlib.pyplot as plt

pixel = np.loadtxt ("Night1/Flat1.csv", delimiter=',',usecols=(0,))
halogen = np.loadtxt ("Night1/Flat1.csv", delimiter=',',usecols=(1,))
flat_dark = np.loadtxt ("Night1/Flat1_dark.csv", delimiter=',',usecols=(1,))
halogen = halogen - flat_dark
halogen = halogen[::-1]
pixel = (4.277*pixel) + 3828.53
#pixel = (3.847*pixel) + 4846.31
    
c = 300000000
h =6.63*(10**(-34))
k = 1.38*(10**(-23))
T = 3200
v = c/(pixel*(10**(-10)))
a = 2*h*(v**3)/(c**2)
b = np.exp(h*v/(k*T))-1
Blackbody =a/b

T2 = 4379
b2 = np.exp(h*v/(k*T2))-1
Blackbody2 =(a/b2)/80

sourcelist = ["Vega", "Enif"]
e = [4, 3]
q = [0.02*(10**(-9)),0.6*(10**(-27)),300]
w = [0.02*(10**(-9)),0.6*(10**(-27)),500]
m = 0
pixellist = []
intensitylist = []
outputlist = []
for source in sourcelist:
    #plt.figure(figsize = (11,8))
    trial = np.arange(1,e[m],1)
    for n in trial:
        pixel = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(0,))
        intensity = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(1,))
        intensity_Dark = np.loadtxt("Night1/{0}0{1}_dark.csv".format(source,n),delimiter=',',usecols=(1,))
        
        intensity = intensity - intensity_Dark
        intensity = intensity [::-1]

        pixel = (4.277*pixel) + 3828.53 # changing pixel to wavelength
        intensity = intensity * Blackbody/halogen
        
        pixellist.append(pixel)
        intensitylist.append(intensity)
        
        centroidpixel = np.zeros(1)
        centroidintensity = np.zeros(1)
        for i in range(len(intensity)-2):
            if intensity[i+1]<intensity[i]:
                i += 1
                if intensity[i+1]>=(intensity[i]+q[m]):
                    if intensity[i-1]>=(intensity[i])+(w[m]):
                        #print intensity[i-2]
                        
                        centroidpixel = np.append(centroidpixel,pixel[i])
                        centroidintensity= np.append(centroidintensity, intensity[i])
        centroidpixel = np.delete(centroidpixel, 0)
        centroidintensity = np.delete(centroidintensity, 0)
        output = np.column_stack((centroidpixel, centroidintensity))
        outputlist.append(output)
    m+=1
'''
intensity_Dark = np.loadtxt("Night1/Vega03_dark.csv",delimiter=',',usecols=(1,))

sourcelist = ["Albero01", "Albero02"]
e = [2, 2]
q = 0.06*(10**(-27))
w = 0.06*(10**(-27))
for source in sourcelist:
    pixel = np.loadtxt("Night1/{0}.csv".format(source),delimiter=',',usecols=(0,))
    intensity = np.loadtxt("Night1/{0}.csv".format(source),delimiter=',',usecols=(1,))
    
        
    intensity = intensity - intensity_Dark
    intensity = intensity [::-1]
        
    #pixel = (0.197*pixel) + 344.31
    pixel = (4.277*pixel) + 3828.53
    intensity = intensity * Blackbody/halogen
    
    pixellist.append(pixel)
    intensitylist.append(intensity)
    
    centroidpixel = np.zeros(1)
    centroidintensity = np.zeros(1)
    for i in range(len(intensity)-2):
        if intensity[i+1]<intensity[i]:
            i += 1
            if intensity[i+1]>=(intensity[i]+q):
                #c = intensity[1000]+4000 # change this value to get even smaller peak
                if intensity[i-1]>=(intensity[i]+w):
                    #cpixel = 0.5*(pixel[i]+pixel[i+1])
                    centroidpixel = np.append(centroidpixel,pixel[i])
                    
                    #cintensity = 0.5*(intensity[i]+intensity[i+1])
                    centroidintensity= np.append(centroidintensity, intensity[i])
    centroidpixel = np.delete(centroidpixel, 0)
    centroidintensity = np.delete(centroidintensity, 0)
    output = np.column_stack((centroidpixel, centroidintensity))
    outputlist.append(output)
'''
'''
######################### plots ###################################
plt.figure(figsize = (11,8))
plt.subplot(4,1,1)
plt.title("Blackbody Spectra of Stars at 60s Exposure Time ")
plt.plot(pixellist[2], intensitylist[2], color='b', label = "Vega")
plt.ylabel("Intensity")
plt.legend(loc =2)

plt.subplot(4,1,2)
plt.plot(pixellist[4], intensitylist[4], color='g', label = "Enif")
plt.ylabel("Intensity")
plt.legend(loc =2)

plt.subplot(4,1,3)
plt.plot(pixellist[5], intensitylist[5], color='r', label = "Albero01")
plt.ylabel("Intensity")
plt.legend(loc =2)

plt.subplot(4,1,4)
plt.plot(pixellist[6], intensitylist[6], color='m', label = "Albero02")
plt.ylabel("Intensity")
plt.legend(loc =2)

#plt.plot(centroidpixel,centroidintensity, 'or' )
plt.xlabel("Wavelength")
plt.tight_layout()

'''

plt.figure(figsize = (10,4))
#plt.plot(pixellist[0], intensitylist[0], color='r', label = "15s")
#plt.plot(pixellist[1], intensitylist[1], color='b', label = "30s")
#plt.plot(centroidpixel,centroidintensity, 'or' )
plt.plot(pixellist[2], intensitylist[2], color='g', label = "60s")
#plt.plot ((pixel), halogen/Blackbody)
#plt.plot (pixel, Blackbody2)
plt.title("Vega at Different Exposure Time")
plt.ylabel("Intensity")
#plt.legend(loc =2 )
plt.xlabel("Wavelength")

plt.text(4000, 5.33*(10**(-10)), 'H-delta')
plt.text(4340 , 0.78*(10**(-10)), 'H-gemma')
plt.text(4800 , 6.6*(10**(-10)), 'H-Beta')
plt.text(6550 , 2.0*(10**(-9)), 'H-alpha')
plt.text(6907 , 2.6*(10**(-9)) , 'O2')

plt.tight_layout()



###################### saving files ##############################
#np.savetxt("centroid_for_{0}{1}.txt".format(source,n), output, fmt='%.2f')

'''
plt.figure ()
plt.plot (pixel, Blackbody)
plt.title("Blackbody of Halogenlamp at 3200K")
plt.ylabel("Intensity")
#plt.legend(loc =2 )
plt.xlabel("Wavelength")
'''
plt.show()
