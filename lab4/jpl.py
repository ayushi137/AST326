import numpy as np
import matplotlib.pyplot as plt


time = [2455946.68646, 2455947.69476, 2455949.73866, 2455950.68528, 2455955.56062]
t = np.array(time) - 2450000

a = [2,57,49.20, 19,14,30.6]
b = [2,58,44.52, 19,16,46.1]
c = [3,0 ,41.30, 19,21,39.6]
d = [3,1 ,37.46, 19,24,3.7 ]
e = [3,7 ,01.64, 19,38,19.4]

list = [a,b,c,d,e]

ra = []
dec = []

for i in list:
    r = 15*(float(i[0]) + float(i[1])/60. + float(i[2])/3600.)*np.pi/180
    d = (float(i[3]) + float(i[4])/60. + float(i[5])/3600.)*np.pi/180
    ra.append(r)
    dec.append(d)


RA = [0.7576194435787118,
      0.76485374461249533,
      0.77942040615003683,
      0.78675307340167622,
      0.8164160277064193]

DEC = [0.33030226510101246,
       0.33233022813591662,
       0.33635651741614675,
       0.3383543132358271,
       0.34623680993397327]

#RA = [ 0.77589496, 0.77991461,0.78845718,0.79249126,0.8161456]

#DEC = [0.33583406,0.33640024,0.33791579,0.33851884, 0.34273346]

ra_new = []
dec_new = []
j = 0
while j < len(RA):
    
    a1 = RA[j]*180/(np.pi*15)
    a2 = a1%1
    a3 = a2*60%1
    a4 = a3*60
    alpha = [a1-a2,a2*60-a3,a4]
    
    d1 = DEC[j]*180/np.pi
    d2 = d1%1
    d3 = d2*60%1
    d4 = d3*60
    delta = [d1-d2,d2*60-d3,d4]
    
    ra_new.append(alpha)
    dec_new.append(delta)
    j+=1


ra_diff = []
dec_diff = []
rd = []
dd = []
k = 0
while k < len(RA):
    difra = abs(ra[k]-RA[k])
    rd.append(difra)
    a1 = difra*180/(np.pi*15)
    a2 = a1%1
    a3 = a2*60%1
    a4 = a3*60
    alpha = [a1-a2,a2*60-a3,a4]
    
    difdec = abs(dec[k]-DEC[k])
    dd.append(difdec)
    d1 = difdec*180/np.pi
    d2 = d1%1
    d3 = d2*60%1
    d4 = d3*60
    delta = [d1-d2,d2*60-d3,d4]
    
    ra_diff.append(alpha)
    dec_diff.append(delta)
    k+=1


plt.figure()
plt.subplot(2,1,1)
plt.plot (t, ra, '--r', label = 'JPL')
plt.plot (t, RA, 'b', label = 'predected')
plt.legend(loc = 2)
plt.ylabel (r'$\alpha$')

plt.subplot(2,1,2)
plt.plot (t, dec, '--r', label = 'JPL')
plt.plot (t, DEC, 'b', label = 'predicted')
plt.ylabel (r'$\delta$')
plt.legend(loc = 2)
plt.xlabel('Julian Day -2450000 ')
plt.show()

plt.figure()
plt.plot (t, rd, 'g')
plt.plot (t, dd, 'g')
plt.show()

