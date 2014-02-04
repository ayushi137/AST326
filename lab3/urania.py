import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import matplotlib.cm as cm

folder = [19,20,22,23]#,28]
#ra = []
#dec = []
#X = []
#Y = []
#astra = []
#astdec=[]
#Pixelx = [1086,1089,1063,1087]
#Pixely = [992,1007,1039,1008]
m=0
for z in folder:
    centrox = np.loadtxt("centroids/{0}centroid_points.txt".format(z), usecols =(0,))
    centroy = np.loadtxt("centroids/{0}centroid_points.txt".format(z), usecols = (1,))
    intensity = np.loadtxt("intensity/{0}intensity.txt".format(z))
    ################################# plotting ###################################
    plt.figure()
    plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 20000, cmap = cm.gray_r, interpolation ='nearest')
    #plt.plot(centrox,centroy, 'ko', mfc = 'none')
    plt.xlim(0,2049)
    plt.ylim(0,2049)
    plt.colorbar()
    # this is how to plot circles on the plot
    i = 0
    while i < (len(centrox)):
        circ  = plt.Circle((centrox[i],centroy[i]),radius = 20, color='m',fill=False)
        plt.gcf()
        plt.gca().add_artist(circ)
        i+=1

    m+=1

plt.show()