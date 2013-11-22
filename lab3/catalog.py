
import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import string as str
import urllib as url
import matplotlib.cm as cm

def unso (radeg,decdeg,fovam): # RA/Dec in decimals degrees/ J2000.0 FOV in arc min.
    #str1 = 'http://webviz.u-strasbg.fr/viz-bin/asu-tsv/?-source=USNO-B1'
    #str2 = '&-c.ra={0:4.6f}&-c.dec={1:4.6f}&-c.bm{2:4.7f}/{3:4.7f}&-out.max=unlimited'.format(radeg,decdeg,fovam,fovam)
    str1 = 'http://webviz.u-strasbg.fr/viz-bin/asu-tsv/?-source=USNO-B1'
    str2 = '&-c.ra={0:4.6f}&-c.dec={1:4.6f}&-c.bm={2:4.7f}/{3:4.7f}&-out.max=unlimited'.format(radeg,decdeg,fovam,fovam)

    str = str1+str2
    #print str
    f = url.urlopen(str)
    s = f.read()
    f.close()
    #print s

    s1 = s.splitlines()
    #print s1
    s1 = s1[45:-1]
    #print 'HELLO WORLD: {0}'.format(s1)
    
    name = np.array([])
    rad = np.array([])
    ded = np.array([])
    rmag = np.array([])
    for k in s1:
        #print k
        kw = k.split('\t')
        name = np.append(name,kw[0])
        rad = np.append(rad, float(kw[1]))
        ded = np.append(ded, float(kw[2]))
        if kw[12] != '     ':
            rmag = np.append(rmag,float(kw[12]))
        else:
            rmag = np.append(rmag, np.nan)
    #print name, rad, ded, rmag
    return name, rad, ded, rmag


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

#############################getting from the catalog###########################
fits1="NGC7331/NGC7331-S001-R001-C001-r.fts"
s1 = pf.open(fits1)

ras = s1[0].header['ra']
des = s1[0].header['dec']

radeg = 15*(float(ras[0:2]) + float(ras[3:5])/60. + float(ras[6:])/3600.)
dsgn = np.sign(float(des[0:3]))
dedeg = float(des[0:3]) + dsgn*float(des[4:6])/60. + dsgn*float(des[7:])/3600.
fovam = 35.0
print radeg, dedeg, fovam
name, rad, ded, rmag = unso(radeg,dedeg, fovam)
w = np.where(rmag < 13.)[0] # # repesent the magnitude (higher  = less) this will change the number of data I plot from the catalog
################## changing to pixel #########################

ao = radeg*np.pi/180
do = dedeg*np.pi/180
a = rad*np.pi/180
d = ded*np.pi/180
X = - ((np.cos(d))*(np.sin(a-ao)))/((np.cos(do))*(np.cos(d))*(np.cos(a-ao))+((np.sin(d))*(np.sin(do))))
Y = - ((np.sin(do)*np.cos(d)*np.cos(a-ao))-(np.cos(do)*np.sin(d)))/((np.cos(do))*(np.cos(d))*(np.cos(a-ao))+((np.sin(d))*(np.sin(do))))

xo = 1024
yo = 1024

x_catalog = 3454*(X/0.018)+xo
y_catalog = 3454*(Y/0.018)+yo

centrox = np.loadtxt("centroid_points.txt", usecols =(0,))
centroy = np.loadtxt("centroid_points.txt", usecols = (1,))

################## plotting #########################
plt.figure()
plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 20000, cmap = cm.gray_r, interpolation ='nearest')
plt.xlim(0,2049)
plt.ylim(0,2049)
plt.colorbar()
plt.plot(x_catalog[w],y_catalog[w],'.')
i = 0
while i < (len(centrox)):
    circ  = plt.Circle((centrox[i],centroy[i]),radius = 20, color='m',fill=False)
    plt.gcf()
    plt.gca().add_artist(circ)
    i+=1

#plt.locator_params(axis='x',nbins=4)
#plt.locator_params(axis='y',nbins=4)
#plt.tick_params('x',pad=10)
#plt.xlabel('RA [Deg]')
#plt.ylabel('Dec [Deg]')
#plt.ticklabel_format(useOffset=False)
#plt.axis('scaled')
#plt.xlim([339.5,339.1])

plt.show()
