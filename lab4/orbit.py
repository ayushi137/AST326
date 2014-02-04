import numpy as np
import matplotlib.pyplot as plt


def find_v(e,E):
    true = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    return true

###19
'''
    [[[[ 189497.51740411]] [[ 1896.21766195]] [[ 1027.26419218]]]
    [[[ 8.3976048]] [[ 194522.62451981]] [[ 1022.29707521]]]
    [0 0 1]]
    
    Chi^2 x: [[ 194.6277209]]
    Chired^2 x: [[ 24.32846511]]
    Chi^2 y: [[ 4.14862438]]
    Chired^2 y: [[ 0.51857805]]
    '''

###20
'''
    [[[[ 191690.91681286]] [[-889.3848706]] [[ 1020.20026665]]]
    [[[ 3134.90897751]] [[ 189796.0701779]] [[ 1018.747944]]]
    [0 0 1]]
    
    Chi^2 x: [[ 194.6277209]]
    Chired^2 x: [[ 24.32846511]]
    Chi^2 y: [[ 4.14862438]]
    Chired^2 y: [[ 0.51857805]]
'''
### 22
'''
    T: [[[[ 191768.93408327]] [[-4067.17277444]] [[ 1028.40092941]]]
    [[[-950.66486019]] [[ 190951.44336709]] [[ 1030.6404094]]]
    [0 0 1]]
    Chi^2 x: [[ 1012.86770911]]
    Chired^2 x: [[ 63.30423182]]
    Chi^2 y: [[ 741.70751153]]
    Chired^2 y: [[ 46.35671947]]
'''


# ra, dec mine
a1 = [ 0.77589496, 0.33583406]
a2 = [ 0.77991461, 0.33640024]
a3 = [ 0.78845718, 0.33791579]
a4 = [ 0.79249126, 0.33851884]
a5 = [ 0.8161456, 0.34273346]


# jpl
a1 = [ 0.7758861189740791,0.33583334578666223]
a2 = [ 0.7799091028999261,0.33649026832456563]
a3 = [ 0.7884015841519219,0.33791319647862217]
a4 = [ 0.7924856546015885,0.33861181299310095]
a5 = [ 0.8149589503825805,0.34257128632672257]


asteroid = [a1, a2, a3, a4, a5]

date = ["20/01/2012", "21/01/2012", "23/01/2012", "24/01/2012", "29/01/2012"]
time = ["04:28:30", "04:40:27", "05:43:40", "04:26:48", "01:27:18"]
J2000 = [2455946.68646, 2455947.69476, 2455949.73866, 2455950.68528, 2455955.56062]
# X, Y, Z
R1 = [-4.852325900266916e-01,  8.563819442309909e-01, -2.676783649804444e-05]
R2 = [-5.005639235484095e-01,  8.476786115155864e-01, -2.723171022212407e-05]
R3 = [-5.311509164664779e-01,  8.292136310803589e-01, -2.794675828254335e-05]
R4 = [-5.450861676152123e-01,  8.202934515047903e-01, -2.812109168370201e-05]
R5 = [-6.143806933050102e-01,  7.707734163615326e-01, -2.700019419225799e-05]
RRR = [R1,R2,R3,R4,R5]

#CONSTANTS
k = 0.017202098950 #AU^3/2 d^-1 ... sqrt(GM)
epislon = 23.43929111*np.pi/180

############## conversion from equatorial to eliptic coordiantes ##########
m = 0
Seq = [] # this will have s1, s2, s3
while m < 5:
    s = []
    s.append(np.cos(asteroid[m][0])*np.cos(asteroid[m][1]))
    s.append(np.sin(asteroid[m][0])*np.cos(asteroid[m][1]))
    s.append(np.sin(asteroid[m][1]))
    Seq.append(s)
    m+=1



Seq = np.array(Seq)
Tx = np.array([[1,0,0],
              [0, np.cos(epislon), np.sin(epislon)],
              [0, -np.sin(epislon), np.cos(epislon)]])

m = 0
S = [] # this will have s1, s2, s3
while m < 5:
    S.append(np.dot(Tx,Seq[m]))
    m+=1

s1 = np.array(S[0])
s2 = np.array(S[1])
s3 = np.array(S[2])
s4 = np.array(S[3])
s5 = np.array(S[4])
sss = [s1,s2,s3,s4,s5]

############################### calculation ##########################

choice = [0,1,2]

tau1 = J2000[choice[1]] - J2000[choice[0]] #@@@@@@@@@@
tau3 = J2000[choice[2]] - J2000[choice[1]] #@@@@@@@@@@

# step 4
# ds/dt for only 2
sd = ((tau3*(sss[1]-sss[0]))/(tau1*(tau1+tau3)))+((tau1*(sss[2]-sss[1]))/(tau3*(tau1+tau3))) #@@@@@@@@@@
# d^2s/dt^2 for only 2
sdd = ((2*(sss[2]-sss[1]))/(tau3*(tau1+tau3)))-((2*(sss[1]-sss[0]))/(tau1*(tau1+tau3))) #@@@@@@@@@@

R = np.array (RRR[1]) #@@@@@@@@@@
RR = np.linalg.norm(R) # RR = sqrt(X^2+Y^2+Z^2)
s = np.array(sss[1]) #@@@@@@@@@@
sd = np.array(sd)
sdd = np.array(sdd)

def rho_calc (k, RR, r,R,s,sd,sdd):
    rho = (k**2)*((1/RR**3)-(1/r**3))*(np.dot(sd,(np.cross(R,s))))/(np.dot(sd,(np.cross(sdd,s))))
    return rho

def r_calc (rho, RR, R, s ):
    r = np.sqrt((rho**2)+(RR**2) + (2*rho*(np.dot(R,s))))
    return r

# this is used to get the values for rho and r
rho = 0.1
r = 0.1
m = 0
rlist = np.zeros(101)
rlist[0] = r
rholist = np.zeros(101)
rholist[0] = rho
while m < 100:
    r = r_calc (rho, RR, R, s )
    rlist[m+1] = r
    rho = rho_calc (k, RR, r,R,s,sd,sdd)
    rholist[m+1] = rho
    m+=1

#r = 2.1082437391712778
plt.figure()
plt.subplot(2,1,1)
g = np.arange(0,101,1)
plt.plot(g, rlist, label = 'r')
plt.ylabel('r')

plt.subplot(2,1,2)
g = np.arange(0,101,1)
plt.plot(g,rholist, label = 'rho')
plt.ylabel(r'$\rho$')
plt.xlabel('Increments')


plt.show()


# drho/dt
rhod = ((k**2)/2)*((1/RR**3)-(1/r**3))*(np.dot(sdd,(np.cross(R,s))))/(np.dot(sdd,(np.cross(sd,s))))

# rr = from sun to asteroid
rr = R + (rho*s)

Rd = ((tau3*(R-RRR[0]))/(tau1*(tau1+tau3)))+((tau1*(RRR[2]-R))/(tau3*(tau1+tau3))) # dR/dt #@@@@@@@@@@
rrd = Rd + rho*sd + rhod*s # drr/dt

##### finding orbital elements
rrdd = rr*(-k**2)/(r**3) # eq 2
hh = np.cross(rr,rrd ) # eq 5
rd = np.linalg.norm(rrd)
h = np.linalg.norm(hh)

V = rd
omega = np.arctan2(-hh[0],hh[1])
omega += np.pi # eq 54
#omega = (2*np.pi)+(omega)
deg = 180/np.pi
#omega = omega*180/np.pi ### works
i = np.arccos(hh[2]/h) # eq 55
#i = i*180/np.pi ### works


a = ((k**2)*r)/((2*(k**2))-(r*(V**2))) ######
e = np.sqrt(1.0-(h**2/(a*(k**2)))) #### eq 56

E = np.arccos((a-r)/(a*e)) # note the sign is depended on the rrd.
M = E - (e*np.sin(E))
v = find_v(e,E) # true anomaly
n = np.sqrt((k**2)/(a**3))
p = 2*np.pi/n
tau = (2455949.73866)-(M/n)
total = np.arccos(((rr[0]*np.cos(omega))+(rr[1]*np.sin(omega)))/r)
w = total - v
ro = r
Eo = E
Mo = M
### values in degree
small_omega = w*deg
Omega = omega*deg
inclination = i*deg
inc = i

print "r: ", r
print "rho: ", rho
print "a: ", a
print "e: ", e
print "i: ", inclination
print "w: ", small_omega
print "Omega: ", Omega
print "tau", tau
################

t = np.arange(tau,tau+p+500, 1) # time interval

M = n*(t - tau) # this is the M from the equation 26 which we call mu

E = np.zeros(len(M)) # this will have Eccentric Anomaly
E[0] = 0


def Mlist (E,e):
    Mhold = E - e*np.sin(E)
    return Mhold

# Mo is the mean anomaly that is depended on E
for i in range(1, len(E)):
    Mo = Mlist(E[i-1],e)
    E[i] = E[i-1] +(M[i] - Mo)/(1-e*np.cos(E[i-1]))

r = a*(1-e*np.cos(E)) # orbital seperation

v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2)) # true anomaly

theta = v + w
# x and y component of orbital seperation
xh = np.zeros(len(theta))
yh = np.zeros(len(theta))

for j in range(1,len(E)):
    xh[j]=(r[j]*np.cos(theta[j]))
    yh[j]=(r[j]*np.sin(theta[j]))

xo = np.delete(xh,0)
yo = np.delete(yh,0)

ta = np.arange(0,2*np.pi,0.01)
xa = a*np.cos(ta)
ya = a*np.sin(ta)

to = t - 2450000

TzTx = [[np.cos(omega),-np.sin(omega)*np.cos(inc),np.sin(omega)*np.sin(inc)],
		[np.sin(omega),np.cos(omega)*np.cos(inc),-np.cos(omega)*np.sin(inc)],
		[0,np.sin(inc),np.cos(inc)]]

TzTx = np.matrix(TzTx)


timehold = [2455946.68646, 2455947.69476, 2455949.73866, 2455950.68528, 2455955.56062]
r_vec = []
r_eq = []
RA = []
DEC = []
for time in timehold:
    saywhat = np.nonzero(t<=time)
    MAnomaly = n*(time - tau)
    EAnomaly = E[saywhat[0][-1]]
    #EAnomaly = Eo+((MAnomaly-Mo)/(1-e*np.cos(Eo)))
    trueAnomaly = find_v(e,EAnomaly)
    theta_predict = trueAnomaly+w
    r_predict = a*(1-e*np.cos(EAnomaly))
    rvec = np.matrix([(r_predict*np.cos(theta_predict)),(r_predict*np.sin(theta_predict)),0])
    recliptic = (TzTx*rvec.T)
    Te = np.matrix([[1,0,0],
                    [0, np.cos(epislon), -np.sin(epislon)],
                    [0, np.sin(epislon), np.cos(epislon)]])
    requatorial = np.array((Te*recliptic).T)
    recliptic = np.array(recliptic.T)
    
    Rvec = np.array((Te*np.matrix(R).T).T)
    rho_s = recliptic - R
    rho_s2 = Te*np.matrix(rho_s).T
    rho_s = np.array(rho_s2.T)
    rho = np.linalg.norm(rho_s)
    
    r_a = np.arctan(rho_s[0][1]/rho_s[0][0])
    dec = np.arcsin(rho_s[0][2]/rho)
    RA.append(r_a)
    DEC.append(dec)
    r_vec.append(rvec)

    
'''

plt.figure()
plt.plot(to,E, 'r', label='Eccentic Anomaly')
plt.plot(to,M, '--b', label='Mean Anomaly')
plt.legend(loc = 2)
plt.ylabel ('Anomoly [radians]')
plt.xlabel ('Julian Day -2450000')
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(to,r)
plt.ylabel('r [AU]')

plt.subplot(2,1,2)
plt.plot(to, v)
plt.ylabel ('v [radians]')
plt.xlabel ('Julian Day -2450000')
#plt.tight_layout()
plt.show()
'''

color_index=['m','y','b','r','c']
r_vec = np.array(r_vec)
plt.figure(figsize= (7,7))
plt.scatter (xo[0], yo[0], color='g')
plt.plot (xo,yo, label = 'orbit of asteroid')
'''
for i in range(len(timehold)):
    plt.scatter (r_vec[i][0][0],r_vec[i][0][1],color=color_index[i])#, label = timehold[i])
#plt.scatter (r_vec[1][0][0],r_vec[1][0][1],color='y')
#plt.scatter (r_vec[2][0][0],r_vec[2][0][1],color='b')
#plt.scatter (r_vec[3][0][0],r_vec[3][0][1],color='r')
#plt.scatter (r_vec[4][0][0],r_vec[4][0][1],color='k')
'''
plt.plot (xa, ya, '--k', label = 'circular orbit')
plt.plot (0,0, '+r')
plt.ylabel ('y [AU]')
plt.xlabel ('x [AU]')
plt.legend()
plt.show()


plt.figure()
plt.plot(timehold, RA)

plt.figure()
plt.plot(timehold, DEC)

plt.show()


