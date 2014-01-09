# this program is used to extract dark, bais and flat

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import matplotlib.cm as cm

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
line = line/flat
line = line[::-1]

plt.figure()
# to plot fits file
plt.imshow(flat, origin = 'lower', vmin= 0.5, interpolation ='nearest')
plt.colorbar()
plt.title ('Corrected Data after Dark and Flat reduction')
plt.xlim(200, 300)
plt.ylim(200,300)
'''
lin=np.zeros(1)
for b in range(len(line[0])):
    lin = np.append(lin,line[b])
#plot the graph

plt.figure(4)
plt.hist (lin, bins=100 ,align = 'mid', rwidth = 0.8)
plt.xlim (20000,80000)
plt.ylim(0, 200)
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')
'''
plt.show()