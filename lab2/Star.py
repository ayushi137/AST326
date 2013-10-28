import numpy as np
import matplotlib.pyplot as plt

pixel = np.loadtxt ("Flat1.csv", delimiter=',',usecols=(0,))
halogen = np.loadtxt ("Flat1.csv", delimiter=',',usecols=(1,))
flat_dark = np.loadtxt ("Flat1_dark.csv", delimiter=',',usecols=(1,))
halogen = halogen - flat_dark
halogen = halogen[::-1]
pixel = (4.277*pixel) + 3811.27
#pixel = (3.847*pixel) + 4846.31
    
c = 300000000
h =6.63*(10**(-34))
k = 1.38*(10**(-23))
T = 3200
v = c/pixel
a = 2*h*(v**3)/(c**2)
b = np.exp(h*v/(k*T))-1
Blackbody =a/b

sourcelist = ["Vega", "Enif"]
e = [4, 3]
q = [0.03*(10**(-26)),400,300]
w = [0.03*(10**(-26)),300,500]
m = 0
for source in sourcelist:
    plt.figure(figsize = (11,8))
    trial = np.arange(1,e[m],1)
    for n in trial:
        pixel = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(0,))
        intensity = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(1,))
        intensity_Dark = np.loadtxt("Night1/{0}0{1}_dark.csv".format(source,n),delimiter=',',usecols=(1,))
        
        intensity = intensity - intensity_Dark
        intensity = intensity [::-1]
        
        #pixel = (0.197*pixel) + 344.31
        pixel = (4.277*pixel) + 3811.27
        intensity = intensity * Blackbody/halogen
        
        centroidpixel = np.zeros(1)
        centroidintensity = np.zeros(1)
        for i in range(len(intensity)-2):
            if intensity[i+1]<intensity[i]:
                i += 1
                if intensity[i+1]>=(intensity[i]+q[m]):
                    #c = intensity[1000]+4000 # change this value to get even smaller peak
                    if intensity[i-1]>=(intensity[i]+w[m]):
                        #cpixel = 0.5*(pixel[i]+pixel[i+1])
                        centroidpixel = np.append(centroidpixel,pixel[i])
                        
                        #cintensity = 0.5*(intensity[i]+intensity[i+1])
                        centroidintensity= np.append(centroidintensity, intensity[i])
        centroidpixel = np.delete(centroidpixel, 0)
        centroidintensity = np.delete(centroidintensity, 0)
        output = np.column_stack((centroidpixel, centroidintensity))
       
        ######################### plots ###################################
        plt.subplot(e[m]-1,1,n)
        #plt.figure()
        plt.plot(pixel, intensity, color='b')
        plt.plot(centroidpixel,centroidintensity, 'or' )
        plt.title("Spectra for {0} {1}".format(source,n))
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.tight_layout()


        ###################### saving files ##############################
    
        #np.savetxt("centroid_for_{0}{1}.txt".format(source,n), output, fmt='%.2f')
    m+=1
'''
plt.figure (n+1)
plt.plot (pixel, Blackbody)
'''

plt.show()
