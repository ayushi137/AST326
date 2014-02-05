limport numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import string as str
import urllib as url
import matplotlib.cm as cm

################################### function to load the catalog ###############

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

########################## functiion to change degree to pixel #################
def deg_to_pixel(w):
    X2 = - ((np.cos(d))*(np.sin(a-ao)))/((np.cos(do))*(np.cos(d))*(np.cos(a-ao))+((np.sin(d))*(np.sin(do))))
    Y2 = - ((np.sin(do)*np.cos(d)*np.cos(a-ao))-(np.cos(do)*np.sin(d)))/((np.cos(do))*(np.cos(d))*(np.cos(a-ao))+((np.sin(d))*(np.sin(do))))
    '''
    print len(X2[w])
    
    plt.figure()
    plt.plot(X2[w],Y2[w], '.')
    '''
    
    
    x_catalog = f*(X2/p)+xo
    y_catalog = f*(Y2/p)+yo
    
    return x_catalog, y_catalog, X2, Y2


########################## functiion to find match with centroids ##############
def finding_match (x_catalog,y_catalog,centrox, centroy):
    
    detectedx = np.array([])
    detectedy = np.array([])
    catalogx = np.array([])
    catalogy = np.array([])
    differncex = np.array([])
    differncey = np.array([])
    x = x_catalog
    y = y_catalog
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
    return detectedx,detectedy, catalogx, catalogy, differncex, differncey

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


#############################getting from the catalog###########################
fits1="NGC7331/NGC7331-S001-R001-C001-r.fts"
s1 = pf.open(fits1)

ras = s1[0].header['ra']
des = s1[0].header['dec']

radeg = 15*(float(ras[0:2]) + float(ras[3:5])/60. + float(ras[6:])/3600.)
dsgn = np.sign(float(des[0:3]))
dedeg = float(des[0:3]) + dsgn*float(des[4:6])/60. + dsgn*float(des[7:])/3600.
fovam = 35.0
print 'radeg', radeg,'dedeg:',  dedeg,'fovam:', fovam
name, rad, ded, rmag = unso(radeg,dedeg, fovam)
w = np.where(rmag < 13.)[0] # # repesent the magnitude (higher  = less) this will change the number of data I plot from the catalog


################## loading centroids #########################

centrox = np.loadtxt("centroid_points.txt", usecols =(0,))
centroy = np.loadtxt("centroid_points.txt", usecols = (1,))

################## deg to pixel #########################
ao = radeg*np.pi/180
do = dedeg*np.pi/180
a = rad[w]*np.pi/180
d = ded[w]*np.pi/180

xo = 1024
yo = 1024
f = 3454
p = 0.018
F = f/p
print 'f/p:',F
x_catalog, y_catalog, X2, Y2 = deg_to_pixel(w)

################## finding the match #########################
detectedx, detectedy, catalogx, catalogy, differncex,differncey = finding_match(x_catalog,y_catalog,centrox,centroy)

X = (catalogx-xo)*p/f
#print len(X)
Y = (catalogy-yo)*p/f
'''
plt.figure()
plt.plot(X,Y, '.')
'''
################## plate scaling #########################
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
c1 = B3*a
c2 = B3*b

print 'c1:',c1
print 'c2:',c2
# constructing T matrix to calculate x = TX
T = np.matrix([[(F*c1[0]), (F*c1[1]),(c1[2])],
               [(F*c2[0]), (F*c2[1]),(c2[2])],
               [0,0,1]])
T2 = np.matrix([[(c1[0]), (c1[1]),(c1[2])],
                [(c2[0]), (c2[1]),(c2[2])],
                [0,0,1]])
detT = np.linalg.det(T)
detT2 = np.linalg.det(T2)

print 'sqrt(T):',np.sqrt(detT)
print 'T:', T



# to get new x and y pixel parameters
bigX = np.zeros([3,len(X)])
for h in range(len(X)):
    bigX[0][h] = X[h]
    bigX[1][h] = Y[h]
    bigX[2][h] = 1
bigX = np.matrix(bigX)
#x = np.zeros(len(X))
#y = np.zeros(len(X))
#xbar = np.matrix([x,y])

# x = TX
x_bar = T*bigX

# to calculate chi square for x
beta = B*c1
omega = a-beta
omegat = omega.transpose()
Chi_2 = omegat*omega
print 'Chi^2 x:', Chi_2
Chi_red = Chi_2/(B.shape[0]-B.shape[1])
print 'Chired^2 x:',Chi_red

beta2 = B*c2
omega2 = b-beta2
omegat2 = omega2.transpose()
Chi_22 = omegat2*omega2
print 'Chi^2 y:',Chi_22
Chi_red2 = Chi_22/(B.shape[0]-B.shape[1])
print 'Chired^2 y:',Chi_red2

################### ccd to radians ###############################
ccdx = np.zeros([3,len(X)])
for h in range(len(X)):
    ccdx[0][h] = detectedx[h]
    ccdx[1][h] = detectedy[h]
    ccdx[2][h] = 1

ccdx = np.matrix(ccdx)
Tinverse = T.getI()
ccdX = Tinverse*ccdx
XX = np.array(ccdX[0])[0]
YY = np.array(ccdX[1])[0]

RA = np.arctan(-XX/(np.cos(do)-(YY*np.sin(do)))) + ao
part1 =(np.sin(do)+(YY*np.cos(do)))
part2 =((1+XX**2+YY**2)**(1/2))
DEC = np.arcsin(part1/part2)

RAcat = np.arctan(-X/(np.cos(do)-(Y*np.sin(do)))) + ao
part21 =(np.sin(do)+(Y*np.cos(do)))
part22 =((1+X**2+Y**2)**(1/2))
DECcat = np.arcsin(part21/part22)


#RA = RA*180/np.pi
#DEC = DEC*180/np.pi
'''
plt.figure()
plt.plot(ccdX[0],ccdX[1],'b.')
plt.plot(bigX[0],bigX[1],'r+')
'''


################## plotting ######################################
'''
plt.figure()
plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 20000, cmap = cm.gray_r, interpolation ='nearest')
plt.xlim(0,2049)
plt.ylim(0,2049)
plt.title ('Detected centroids')
plt.colorbar()
#plt.plot(x_catalog,y_catalog,'.')
i = 0
# plotting centroids
while i < (len(centrox)):
    circ  = plt.Circle((centrox[i],centroy[i]),radius = 18, color='m',fill=False)
    plt.gcf()
    plt.gca().add_artist(circ)
    i+=1


plt.figure(figsize = (9.5,5))
plt.subplot(1,2,1)
plt.plot(rad[w],ded[w],'g.')
plt.locator_params(axis='x',nbins=4)
plt.locator_params(axis='y',nbins=4)
plt.tick_params('x',pad=10)
plt.xlabel('RA [Deg]')
plt.ylabel('Dec [Deg]')
plt.ticklabel_format(useOffset=False)
plt.axis('scaled')
plt.xlim([339.5,339.1])
plt.title('Data from the Catalog')

plt.subplot(1,2,2)
plt.plot(X2, Y2, 'b.')
plt.title('Projected Coordinates')
plt.xlabel('X')
plt.ylabel('Y')

#~~~~~~~~~~~~~~~~
plt.figure()
#plt.imshow(intensity, origin = 'lower', vmin = 0, vmax= 20000, cmap = cm.gray_r, interpolation ='nearest')
plt.xlim(0,2049)
plt.ylim(0,2049)
#plt.colorbar()
plt.plot(x_catalog,y_catalog,'g*', label = 'USNO')
plt.plot(centrox, centroy, 'r.', label = 'CCD')
plt.title('Overlapping of Centroids and Catalog')
plt.xlabel('x[pixel]')
plt.ylabel('y[pixel]')
plt.legend(loc = 2)

#~~~~~~~~~~~~~~~~
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.plot (catalogx, catalogy, 'b.', label = 'USNO')
plt.plot (detectedx, detectedy, 'r+', label = 'CCD')
plt.title('Matched Centroids and Catalog')
plt.xlabel('x[pixel]')
plt.ylabel('y[pixel]')
plt.legend(loc = 2)

#~~~~~~~~~~~~~~~~
plt.subplot(1,2,2)
plt.plot(detectedx,differncex,'+', label = 'x')
plt.plot(detectedy,differncey,'.', label = 'y')
plt.title('Pixel offset between Centroids and Catalog')
plt.xlabel('x or y[pixel]')
plt.ylabel('pixel offset')
plt.legend(loc = 2)
'''
#~~~~~~~~~~~~~~~~~

test1 = np.array(x_bar[0])
test2 = np.array(x_bar[1])
test = []
testing = []
for i in range(len(test1[0])):
    test.append(float(test1[0][i]))
    testing.append(float(test2[0][i]))

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.plot(test,testing, 'b.', label = 'USNO')
plt.plot (detectedx, detectedy, 'r+', label = 'CCD')
plt.title('After plate constant correction')
plt.xlabel('x[pixel]')
plt.ylabel('y[pixel]')
plt.legend(loc = 2)

#~~~~~~~~~~~~~~~
q = np.matrix(detectedx)
r = np.matrix(detectedy)
p = x_bar[0] - q
o = x_bar[1] - r
test1 = np.array(p[0])
test2 = np.array(o[0])

test = []
testing = []
for i in range(len(test1[0])):
    test.append(float(test1[0][i]))
    testing.append(float(test2[0][i]))

plt.subplot(1,2,2)
plt.plot(detectedx,test, 'b+',label = 'x')
plt.plot(detectedy,testing, 'g.',label = 'y')
plt.title('Pixel offset between Centroids and Catalog')
plt.xlabel('x or y[pixel]')
plt.ylabel('pixel offset')
plt.legend(loc = 2)

#~~~~~~~~~~~~~~~~

plt.figure()
plt.plot(RA, DEC, 'r+')
plt.plot(RAcat, DECcat, 'b.')


q = np.matrix(detectedx)
r = np.matrix(detectedy)
p = x_bar[0] - q
o = x_bar[1] - r
test1 = np.array(p[0])
test2 = np.array(o[0])

test = []
testing = []
for i in range(len(test1[0])):
    test.append(float(test1[0][i]))
    testing.append(float(test2[0][i]))

hold =[]
hold2 = []
x = []
y = []
for i in range(len(test)):
    if test[i] <=2 and test[i]>= -2:
        hold.append(test[i])
        x.append(detectedx[i])
    if testing[i] <=2 and testing[i]>= -2:
        hold2.append(testing[i])
        y.append(detectedy[i])

meanx = np.average(hold)
sigmax = np.std(hold)
print 'mean of offset x:',meanx,'error in x', sigmax
meany = np.average(hold2)
sigmay = np.std(hold2)
print 'mean of offset y:',meany, 'error in y',sigmay

plt.figure()
plt.plot(x,hold, 'b+',label = 'x')
plt.plot(y,hold2, 'g.',label = 'y')
plt.title('Pixel offset between Centroids and Catalog')
plt.xlabel('x or y[pixel]')
plt.ylabel('pixel offset')

plt.legend(loc = 2)



plt.show()
#plt.locator_params(axis='x',nbins=4)
#plt.locator_params(axis='y',nbins=4)
#plt.tick_params('x',pad=10)
#plt.xlabel('RA [Deg]')
#plt.ylabel('Dec [Deg]')
#plt.ticklabel_format(useOffset=False)
#plt.axis('scaled')
#plt.xlim([339.5,339.1])
