
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
intensity = intensity[::-1] # the actual data

################## loading centroids #########################

centrox = np.loadtxt("centroid_points.txt", usecols =(0,))
centroy = np.loadtxt("centroid_points.txt", usecols = (1,))

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

xo = 1024
yo = 1024
f = 3454
p = 0.018

#def deg_to_pixel():
ao = radeg*np.pi/180
do = dedeg*np.pi/180
print ao
print do
a = rad*np.pi/180
d = ded*np.pi/180
X2 = - ((np.cos(d))*(np.sin(a-ao)))/((np.cos(do))*(np.cos(d))*(np.cos(a-ao))+((np.sin(d))*(np.sin(do))))
Y2 = - ((np.sin(do)*np.cos(d)*np.cos(a-ao))-(np.cos(do)*np.sin(d)))/((np.cos(do))*(np.cos(d))*(np.cos(a-ao))+((np.sin(d))*(np.sin(do))))

#print len(X2[w])

#plt.figure()
#plt.plot(X2[w],Y2[w], '.')



x_catalog = f*(X2/p)+xo
y_catalog = f*(Y2/p)+yo


################## finding the match #########################

detectedx = np.array([])
detectedy = np.array([])
catalogx = np.array([])
catalogy = np.array([])
differncex = np.array([])
differncey = np.array([])
x = x_catalog[w]
y = y_catalog[w]

for q in range(len(centrox)):
    x_max = centrox[q]
    #print x_max
    y_max = centroy[q]
    #print y_max
    
    for i in range(len(x)):
        dx = (x[i]- x_max)
        dy = (y[i] - y_max)
        R = np.sqrt(abs(dx**2)+abs(dy**2))
    
        if R < 18:
            differncex = np.append(differncex,dx)
            differncey = np.append(differncey,dy)
            detectedx = np.append(detectedx,x_max)
            detectedy = np.append(detectedy,y_max)
            catalogx = np.append(catalogx,x[i])
            catalogy = np.append(catalogy,y[i])

X = (catalogx-xo)*p/f
#print len(X)
Y = (catalogy-yo)*p/f

#plt.figure()
#plt.plot(X,Y, '.')
################## plate scaling #########################
F = f/p

B = np.zeros([len(catalogx),3])
a = np.zeros([len(catalogx),1])
b = np.zeros([len(catalogx),1])
for i in range(len(catalogx)):
    B[i][0] = (F*X[i])
    B[i][1] = (F*Y[i])
    B[i][2] = 1
    a[i][0] = detectedx[i]
    b[i][0] = detectedy[i]

##### to calcualte c and d values (these are plate constants)
B = np.matrix(B)
Bt = B.transpose()
B2 = Bt*B
Binverse= B2.getI()
B3 = Binverse*Bt
c = B3*a
d = B3*b


# constructing T matrix to calculate x = TX
T = np.matrix([[(F*c[0]), (F*c[1]),(c[2])],
               [(F*d[0]), (F*d[1]),(d[2])],
               [0,0,1]])
T2 = np.matrix([[(c[0]), (c[1]),(c[2])],
                [(d[0]), (d[1]),(d[2])],
                    [0,0,1]])
detT = np.linalg.det(T)
detT2 = np.linalg.det(T2)



# to get new x and y pixel parameters
bigX = np.zeros([3,len(X)])
for h in range(len(X)):
    bigX[0][h] = X[h]
    bigX[1][h] = Y[h]
    bigX[2][h] = 1
bigX = np.matrix(bigX)
x = np.zeros(len(X))
y = np.zeros(len(X))
xbar = np.matrix([x,y])

# x = TX
x_bar = T*bigX

# to calculate chi square
beta = B*c
omega = a-beta
omegat = omega.transpose()
Chi_2 = omegat*omega
print Chi_2
Chi_red = Chi_2/(B.shape[0]-B.shape[1])
print Chi_red



################## plotting #########################
'''
plt.figure()
plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 20000, cmap = cm.gray_r, interpolation ='nearest')
plt.xlim(0,2049)
plt.ylim(0,2049)
plt.colorbar()
plt.plot(x_catalog[w],y_catalog[w],'.')
i = 0
# plotting centroids
while i < (len(centrox)):
    circ  = plt.Circle((centrox[i],centroy[i]),radius = 18, color='m',fill=False)
    plt.gcf()
    plt.gca().add_artist(circ)
    i+=1

#~~~~~~~~~~~~~~~~
plt.figure()
plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 20000, cmap = cm.gray_r, interpolation ='nearest')
plt.xlim(0,2049)
plt.ylim(0,2049)
plt.colorbar()
plt.plot(x_catalog[w],y_catalog[w],'.')
plt.plot(centrox, centroy, 'g.')
'''
#~~~~~~~~~~~~~~~~
plt.figure()
plt.plot (catalogx, catalogy, 'b.')
plt.plot (detectedx, detectedy, 'r+')

#~~~~~~~~~~~~~~~~
plt.figure()
plt.plot(detectedx,differncex,'+')
plt.plot(detectedy,differncey,'.')


plt.figure()
plt.plot(x_bar[0],x_bar[1], 'b.')
plt.plot (detectedx, detectedy, 'r+')


plt.show()
