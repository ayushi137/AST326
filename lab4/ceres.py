import numpy as np
import matplotlib.pyplot as plt

#a, omega, i, e, w, tau
true = [2.766, 80.72*np.pi/180, 10.61*np.pi/180, 0.079, 73.12*np.pi/180, 2454868]
estimated = [2.947, 80.65*np.pi/180, 10.56, 0.079, 63.20*np.pi/180, 2454833]
error = [-0.18, 0.07*np.pi/180, 0.05, -0.05, 9.9*np.pi/180, 35]

# day, lambda, beta, X, Y, Z
UT1 = [2454702.5, 121.7592648*np.pi/180, 4.0625653*np.pi/180, 0.8849686471, -0.4888489729, 4.466373306*(10**(-6))]
UT2 = [2454703.5, 122.1865441*np.pi/180, 4.0992581*np.pi/180, 0.8928865393, -0.4737871683, 4.402701086*(10**(-6))]
UT3 = [2454704.5, 122.6133849*np.pi/180, 4.1361592*np.pi/180, 0.9005490495, -0.4585878955, 4.483801584*(10**(-6))]
UT = [UT1, UT2, UT3]


##### key step 1 #######

a = true[0] # Semimajor axis
tau = true[5] # epoch of Perihelion
k = 0.017202098950 #AU^3/2 d^-1 ... sqrt(GM)
n = np.sqrt((k**2)/(a**3))
p = 2*np.pi/n # period
t = np.arange(2454868,2454868+p+ 500, 0.1) # time interval
e = true[3] # eccentricity
w = true[4] # argument of perihelion

M = n*(t - tau) # this is the M from the equation 26 which we call mu

E = np.zeros(len(M)) # this will have Eccentric Anomaly
E[0] = M[0]


def Mlist (E):
    Mhold = E - (e*np.sin(E))
    return Mhold

# Mo is the mean anomaly that is depended on E
for i in range(1, len(E)):
    Mo = Mlist(E[i-1])
    E[i] = E[i-1] +((M[i] - Mo)/(1-(e*np.cos(E[i-1]))))

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

plt.figure()
plt.plot(to,E, 'r', label='Eccentic Anomaly')
plt.plot(to,M, '--b', label='Mean Anomaly')
plt.legend(loc = 2)
plt.ylabel ('Anomoly [radians]')
plt.xlabel ('Julian Day -2450000')

plt.figure()
plt.subplot(2,1,1)
plt.plot(to,r)
plt.ylabel('r [AU]')

plt.subplot(2,1,2)
plt.plot(to, v)
plt.ylabel ('v [radians]')
plt.xlabel ('Julian Day -2450000')

plt.figure(figsize = (7,7))
plt.scatter (xo[0], yo[0], color='g')
plt.plot (xo,yo, label = 'orbit of asteroid')
plt.plot (0,0, '+r')
plt.plot (xa, ya, '--r', label = 'circualar orbit')
plt.ylabel ('y [AU]')
plt.xlabel ('x [AU]')
plt.legend()

##### key term 3, 4, 5, 6, 7 #######
m = 0
S = [] # this will have s1, s2, s3
while m < 3:
    s = []
    s.append(np.cos(UT[m][1])*np.cos(UT[m][2]))
    s.append(np.sin(UT[m][1])*np.cos(UT[m][2]))
    s.append(np.sin(UT[m][2]))
    S.append(s)
    m+=1

tau1 = UT2[0] - UT1[0]
tau3 = UT3[0] - UT2[0]

s1 = np.array(S[0])
s2 = np.array(S[1])
s3 = np.array(S[2])

# ds/dt for only 2
sd = ((tau3*(s2-s1))/(tau1*(tau1+tau3)))+((tau1*(s3-s2))/(tau3*(tau1+tau3)))
# d^2s/dt^2 for only 2
sdd = ((2*(s3-s2))/(tau3*(tau1+tau3)))-((2*(s2-s1))/(tau1*(tau1+tau3)))

R = [UT2[3], UT2[4], UT2[5]] # X Y Z of second epoch
R = np.array(R)
RR = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2) # RR = sqrt(X^2+Y^2+Z^2)
s = np.array(s2)
sd = np.array(sd)
sdd = np.array(sdd)

def rho_calc (k, RR, r,R,s,sd,sdd):
    rho = (k**2)*((1/RR**3)-(1/r**3))*(np.dot(sd,(np.cross(R,s))))/(np.dot(sd,(np.cross(sdd,s))))
    return rho

def r_calc (rho, RR, R, s ):
    r = np.sqrt((rho**2)+(RR**2) + (2*rho*(np.dot(R,s))))
    return r

# this is used to get the values for rho and r
rho = 1
r = 1
m = 0
rlist = np.zeros(21)
rlist[0] = r
rholist = np.zeros(21)
rholist[0] = rho
while m < 20:
    r = r_calc (rho, RR, R, s )
    rlist[m+1] = r
    rho = rho_calc (k, RR, r,R,s,sd,sdd)
    rholist[m+1] = rho
    m+=1

g = np.arange(0,21,1)
#plt.plot(g, rlist)
#plt.plot(g,rholist)

# drho/dt
rhod = ((k**2)/2)*((1/RR**3)-(1/r**3))*(np.dot(sdd,(np.cross(R,s))))/(np.dot(sdd,(np.cross(sd,s))))

# rr = from sun to asteroid
rr = R + (rho*s) # this is the vector where as the length is r calculated earlier

R1 = [UT1[3], UT1[4], UT1[5]]
R3 = [UT3[3], UT3[4], UT3[5]]
Rd = ((tau3*(R-R1))/(tau1*(tau1+tau3)))+((tau1*(R3-R))/(tau3*(tau1+tau3))) # dR/dt
rrd = Rd + rho*sd + rhod*s # drr/dt


##### finding orbital elements
rrdd = rr*(-k**2)/(r**3) # eq 2
hh = np.cross(rr,rrd ) # eq 5
rd = np.sqrt(rrd[0]**2 + rrd[1]**2 + rrd[2]**2)
h = np.sqrt(hh[0]**2 + hh[1]**2 + hh[2]**2)
h = np.linalg.norm(hh)

V = rd
omega = np.arctan(-hh[0]/hh[1]) # eq 54
deg = 180/np.pi
#omega = omega*180/np.pi ### works
i = np.arccos(hh[2]/h) # eq 55
#i = i*180/np.pi ### works

#a = 2.947
a = ((k**2)*r)/((2*(k**2))-(r*(V**2))) ######

e = np.sqrt(1-(h**2/(a*(k**2)))) #### eq 56

E = -np.arccos((a-r)/(a*e)) # note the sign is depended on the rrd.
M = E - (e*np.sin(E))
v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2)) # true anomaly
n = np.sqrt((k**2)/(a**3))
p = 2*np.pi/n
tau2 = (2454703.5)-(M/n)
total = np.arccos(((rr[0]*np.cos(omega))+(rr[1]*np.sin(omega)))/r)
om = total - v
small_omega = om*180/np.pi
Omega = omega*deg
inclination = i*deg

# for lab report we need the gradiant of function rr to show why this sign thing is messed up.

plt.show()