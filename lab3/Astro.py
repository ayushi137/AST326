import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import matplotlib.cm as cm

folder = [19,20,22,23,28]
ra = []
dec = []
X = []
Y = []
astra = []
astdec=[]
Pixelx = [1086,1089,1063,1087,1071.1359772607466]
Pixely = [992,1007,1039,1008,1024.442188922701]
m=0
for z in folder:
    print z
    x = np.loadtxt("{0}centroids.txt".format(z), usecols =(0,))
    y = np.loadtxt("{0}centroids.txt".format(z), usecols = (1,))
    intensity = np.loadtxt("{0}intensity.txt".format(z))

    background = 8000
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
            print " ",
        q+=1

    output = np.column_stack((centrox, centroy))
    X.append(centrox)
    Y.append(centroy)
    np.savetxt ("{0}centroid_points.txt".format(z), output, fmt='%.1i')


    ################### ccd to radians ###############################
    T = np.loadtxt("plateConstants.txt")
    T = np.matrix(T)
    fits1="{0}/30Urania-S001-R001-C001-r.fts".format(z)
    s1 = pf.open(fits1)
    
    ras = s1[0].header['ra']
    des = s1[0].header['dec']
    
    radeg = 15*(float(ras[0:2]) + float(ras[3:5])/60. + float(ras[6:])/3600.)
    dsgn = np.sign(float(des[0:3]))
    dedeg = float(des[0:3]) + dsgn*float(des[4:6])/60. + dsgn*float(des[7:])/3600.
    
    
    ccdx = np.zeros([3,len(centrox)])
    asteroid = np.array([[Pixelx[m]],[Pixely[m]],[1]])

    for h in range(len(centrox)):
        ccdx[0][h] = centrox[h]
        ccdx[1][h] = centroy[h]
        ccdx[2][h] = 1
    
    ao = radeg*np.pi/180
    do = dedeg*np.pi/180 
    ccdx = np.matrix(ccdx)
    asteroid = np.matrix(asteroid)
    Tinverse = T.getI()
    ccdX = Tinverse*ccdx
    Asteroid = Tinverse*asteroid
    XX = np.array(ccdX[0])[0]
    YY = np.array(ccdX[1])[0]
    
    astrox = np.array(Asteroid[0])[0]
    astroy = np.array(Asteroid[1])[0]

    RA = np.arctan(-XX/(np.cos(do)-(YY*np.sin(do)))) + ao
    RAA = (np.arctan(-astrox/(np.cos(do)-(astroy*np.sin(do)))) + ao)*180/np.pi

    part1 =(np.sin(do)+(YY*np.cos(do)))
    part2 =((1+XX**2+YY**2)**(1/2))
    DEC = np.arcsin(part1/part2)
    DECA = (np.arcsin((np.sin(do)+(astroy*np.cos(do)))/((1+astrox**2+astroy**2)**(1/2))))*180/np.pi
    
    RA = RA*180/np.pi
    DEC = DEC*180/np.pi
    
    ra.append(RA)
    dec.append(DEC)
    astra.append(RAA)
    astdec.append(DECA)
    m+=1
    
    
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



colorindex = ['k.','g.', 'b.', 'r.', 'm.']
plt.figure()
for i in range(len(ra)):
    
    #plt.plot(X[i], Y[i], colorindex[i], label = '{0}'.format(folder[i]) )
    #plt.plot(Pixelx[i], Pixely[i], color = 'c', marker = '*', label = 'asteroid' )
    plt.plot(ra[i], dec[i], colorindex[i], label = '{0}'.format(folder[i]+1) )
    plt.plot(astra[i],astdec[i], colorindex[i], marker = '*', markersize =9)#, label = '{0}'.format(folder[i]+1) )#'asteroid' )
    plt.legend(loc =4)
plt.xlabel('RA [Deg]')
plt.ylabel('Dec [Deg]')
plt.title ('Asteroid over 5 day period')
#plt.ylim(19.22, 19.42)


t = [87117,263710,345498,766728]
time = t[2]
r = np.array(astra)
d = np.array(astdec)
r1 = r[0]
r2 = r[3]

d1=d[0]
d2=d[3]

distance = np.sqrt(((r2-r1)**2)+((d2-d1)**2))
print 'distance:', distance

pr = (r2-r1)/time
prs =pr*3600
print prs

pd =(d2-d1)/time
pds = pd*3600
print pds

proper_rad = distance/time
proper_arcsec = proper_rad*3600
print 'proper motion:', proper_rad, '   ', proper_arcsec



motion = [0.00959079,0.00990574,0.0100837]
motionx = [0.00943232,0.0097411,0.00993652]
motiony = [0.00173627,0.0017985,0.00171653]

plt.show()


