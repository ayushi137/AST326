import numpy as np
from matplotlib import pyplot as plt

#first find centroids

calibrationlist = ["NEON"]
triallist = [1,2,3,4,5,6]

centroidpixellist = []
centroidintensitylist = []

for calibration in calibrationlist:
    for number in triallist:
        file = open("/h/ungrad1/berard/Lab2/100ms"+calibration+str(number)+".txt")
        all_lines = file.readlines()
        Pixellist = []
        Intensitylist = []
        for j in range(len(all_lines)-1):
            Pixel = float(all_lines[j].split()[0])
            Intensity = float(all_lines[j].split()[1])
            Pixellist.append(Pixel)
            Intensitylist.append(Intensity)
        for i in range(len(Intensitylist)-2):
            if Intensitylist[i+1]>Intensitylist[i]:
                i += 1
                if Intensitylist[i+1]<Intensitylist[i]:
                    if Intensitylist[i]>10000.0: #adjust this to filter out smaller peaks
                        centroidpixel = 0.5*(Pixellist[i]+Pixellist[i+1])
                        centroidpixellist.append(centroidpixel)
                        centroidintensity = 0.5*(Intensitylist[i]+Intensitylist[i+1])
                        centroidintensitylist.append(centroidintensity)
        print "Source= ",calibration, "Trial= ", number
        print "Centroids found at...", centroidpixellist
        Datasetx = np.array(Pixellist)
        Datasety = np.array(Intensitylist)
        plt.plot(Datasetx,Datasety)
        plt.plot(centroidpixellist,centroidintensitylist,'o')
        plt.show()
        
        
