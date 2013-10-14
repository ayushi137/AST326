import numpy as np
import matplotlib.pyplot as plt

#first find centroids

sourcelist = ["NEON"]
for source in sourcelist:
    trial = np.arange(1,7,1)
    for n in trial:
        pixel = np.loadtxt("100ms{0}{1}.txt".format(source,n),usecols=(0,))
        intensity = np.loadtxt("100ms{0}{1}.txt".format(source,n),usecols=(1,))
        centroidpixel = np.zeros(1)
        centroidintensity = np.zeros(1)
        for i in range(len(intensity)-2):
            if intensity[i+1]>intensity[i]:
                i += 1
                if intensity[i+1]<=intensity[i]:
                    c = intensity[1000]+4000 # change this value to get even smaller peak
                    if intensity[i]>c:
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
        np.savetxt("centroid_for_{0}{1}.txt".format(source,n), output, fmt='%.2f')

plt.show()
