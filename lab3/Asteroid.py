
import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import matplotlib.cm as cm

folder = [19,20,22,23,28]


Intensity = []
Flat = []
Dark = []
RA = []
DEC = []

############################# getting data ###################################
for z in folder:
    Data="{0}/30Urania-S001-R001-C001-r.fts".format(z)
    D="{0}/Dark-S001-R001-C001-B2.fts".format(z)
    F="{0}/AutoFlat-Dusk-r-Bin2-001.fts".format(z)

    source = pf.open(Data)
    d = pf.open(D)
    f = pf.open(F)
    head_source = pf.getheader(Data)
    RA.append(head_source['ra'])
    DEC.append(head_source['dec'])

    line = source[0].data
    dark = d[0].data
    flat = f[0].data
    line = line - dark
    flat = flat/(np.median(flat))
    intensity = line/flat
    intensity = intensity[::-1]
    Intensity.append(intensity)
    Flat.append(flat)
    Dark.append(dark)
    
    background = 5000

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
    np.savetxt ("{0}centroids.txt".format(z), output, fmt='%.3f')
    np.savetxt ("{0}intensity.txt".format(z), intensity, fmt='%.3f')


    plt.figure()
    plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 5000, cmap = cm.gray_r, interpolation ='nearest')
    plt.xlim(0,2049)
    plt.ylim(0,2049)
    plt.colorbar()







plt.show()