#### telescope spectra

import numpy as np
import matplotlib.pyplot as plt



sourcelist = ["Vega"]#, "Erif", "Albero"]
a = [3]#,2,2]
m=1
for source in sourcelist:
    pixellist_D = []
    intensitylist_D = []
    pixellist = []
    intensitylist = []
    trial = np.arange(1,a[m-1]+1,1)
    plt.figure(m)
    plt.suptitle("Data for {0}\n\n\n".format(source))
    for n in trial:
        pixel_Dark = np.loadtxt("Night1/{0}0{1}_dark.csv".format(source,n),delimiter=',',usecols=(0,))
        pixellist_D.append(pixel_Dark)
        intensity_Dark = np.loadtxt("Night1/{0}0{1}_dark.csv".format(source,n),delimiter=',',usecols=(1,))
        intensitylist_D.append(intensity_Dark)
        pixel = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(0,))
        pixellist.append(pixel)
        intensity = np.loadtxt("Night1/{0}0{1}.csv".format(source,n),delimiter=',',usecols=(1,))
        intensitylist.append(intensity)
        
        intensity2 = intensity - intensity_Dark
        plt.subplot(3,1,n)
        plt.plot(pixel, intensity, color='r')
        plt.plot(pixel, intensity_Dark, color='g')
        plt.plot(pixel, intensity2, color='b')
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")

    '''
    if m <= 2:
        intensityarray = np.array(intensitylist)
        intcol = intensityarray.T
        Intensity = np.zeros(len(pixellist[0]))
        for i in range(len(intcol)):
            Intensity[i] = (np.average(intcol[i]))

        plt.subplot(2,2,n+1)
        plt.plot(pixellist[0], Intensity, color='r')
    '''
    m+=1
    plt.tight_layout()

plt.show()
