import numpy as np
import matplotlib.pyplot as plt

#first find centroids

sourcelist = ["Vega"]
a = [3]
for source in sourcelist:
    trial = np.arange(1,2,1)
    for n in trial:
        
        pixel = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(0,))
        intensity = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(1,))
        intensity_Dark = np.loadtxt("Night1/{0}0{1}_dark.csv".format(source,n),delimiter=',',usecols=(1,))
        
        centroidpixel = np.zeros(1)
        centroidintensity = np.zeros(1)
        for i in range(len(intensity)-2):
            if intensity[i+1]<intensity[i]:
                i += 1
                if intensity[i+1]>=(intensity[i]+600):
                    #c = intensity[1000]+4000 # change this value to get even smaller peak
                    if intensity[i-1]>=(intensity[i]+100):
                        #cpixel = 0.5*(pixel[i]+pixel[i+1])
                        centroidpixel = np.append(centroidpixel,pixel[i])
                        
                        #cintensity = 0.5*(intensity[i]+intensity[i+1])
                        centroidintensity= np.append(centroidintensity, intensity[i])
        centroidpixel = np.delete(centroidpixel, 0)
        centroidintensity = np.delete(centroidintensity, 0)
        output = np.column_stack((centroidpixel, centroidintensity))
       
        ######################### plots ###################################
        plt.figure(n)
        plt.plot(pixel, intensity, color='b')
        plt.plot(centroidpixel,centroidintensity, 'or' )
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")

        ###################### saving files ##############################
#np.savetxt("centroid_for_{0}{1}.txt".format(source,n), output, fmt='%.2f')

plt.show()
