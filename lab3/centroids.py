# this program is used to get centroids 

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import matplotlib.cm as cm

############################# getting data ###################################
Data="NGC7331/NGC7331-S001-R001-C001-r.fts"
D="NGC7331/Dark-S001-R003-C003-B2.fts"
F="NGC7331/combflatr.fits"

source = pf.open(Data)
d = pf.open(D)
f = pf.open(F)
head_source = pf.getheader(Data)
head_dark = pf.getheader(D)
head_flat = pf.getheader(F)

line = source[0].data
dark = d[0].data
flat = f[0].data
line = line - dark
flat = flat/(np.median(flat))
intensity = line/flat
intensity = intensity[::-1]
################################# centroids ##################################
background = 15000

x= []
y = []
#for i in range(1,2047):
#for j in range(1,2047):

i = 1
while i < len(intensity)-1:
    j = 1
    while j < len(intensity)-1:
        if intensity[i][j] >= intensity[i-1][j] and intensity[i][j] >= intensity[i+1][j] and intensity[i][j]>=intensity[i][j-1] and intensity[i][j]>= intensity[i][j+1] and intensity[i][j]>=background:
            y.append(i)
            x.append(j)
            #print i , j
        j +=1
    i+=1



output = np.column_stack((x, y))
np.savetxt ("centroids.txt", output, fmt='%.1i')
'''
x = np.loadtxt("centroids.txt", usecols =(0,))
y = np.loadtxt("centroids.txt", usecols = (1,))
'''
# Note for intensity the x and y values are flipped
centrox = np.array([])
centroy = np.array([])
q = 0
while q < len(x):
    x_max = y[q]
    #print x_max
    y_max = x[q]
    #print y_max
    #print intensity[x_max][y_max]
    x_sum = 0
    y_sum = 0
    deno = 0
    box = np.arange(-25, 25, 1)
    box2 = np.arange(-25,25, 1)
    try:
        for i in box:
            for j in box2:
                #print x_max+i, y_max+j
                if (intensity[x_max+i][y_max+j])>=background:
                    #print (intensity[x_max+i][y_max+j])
                    x_sum += (x_max+i)*(intensity[x_max+i][y_max+j])
                    deno += (intensity[x_max+i][y_max+j])
                    y_sum += (y_max+j)*(intensity[x_max+i][y_max+j])
        xavg = x_sum/deno
        yavg = y_sum/deno

        centrox = np.append(centrox, yavg)
        centroy = np.append(centroy, xavg)
        #print centrox, centroy
    except IndexError:
        print
    q+=1

output = np.column_stack((centrox, centroy))
np.savetxt ("centroid_points.txt", output, fmt='%.1i')
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

'''
plt.figure()
plt.imshow(centroids, origin = 'lower', vmin = 0, vmax = 20000, cmap = cm.gray_r, interpolation ='nearest')
plt.colorbar()
'''
plt.show()