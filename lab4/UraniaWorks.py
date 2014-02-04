import numpy as np
import matplotlib.pyplot as plt
#from astropy.time import Time

array = np.array

rad = np.pi/180.

k = 0.01720209895

expected_r_Jan23=[5.557860055930863E-01,2.046029478081670E+00,6.190352100780391E-02]

#FUNCTIONS 

def rho(k,R,s,sdot,sdouble,r):
	numerator = np.dot(sdot,np.cross(R,s))
	denominator = np.dot(sdot,np.cross(sdouble,s))
	Rmag = np.linalg.norm(R)
	constants = (k**2)*((1./Rmag**3)-(1./r**3))
	return constants*(numerator/denominator)

def rhodot(k,R,s,sdot,sdouble,r):
	numerator = np.dot(sdouble,np.cross(R,s))
	denominator = np.dot(sdouble,np.cross(sdot,s))
	Rmag = np.linalg.norm(R)
	constants = ((k**2)/2)*((1./Rmag**3)-(1./r**3))
	return constants*(numerator/denominator)

def radius(rhoval,R,s):
	Rmag = np.linalg.norm(R)
	return np.sqrt((rhoval**2)+(Rmag**2)+2*rhoval*np.dot(R,s))

def trueanomaly(eccentricity,angles):
	return 2*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(angles/2))

def meananomalyE(E,e):
	return E-e*np.sin(E)

def ellipseradius(angles,semimajor,eccentricity):
	return semimajor*(1.0-eccentricity*np.cos(angles))

def meananomalytime(n,tau,t):
	return n*(t-tau)

def circular(angles,radius):
	x = radius*np.cos(angles)
	y = radius*np.sin(angles)
	return x,y


#FIXED PLATE CONSTANTS
alpha = np.array([ 44.45476346,  44.68517463,  45.16998319,  45.40544173,  46.74507317])
delta = np.array([ 19.22012634,  19.27927308,  19.36130666,  19.40179464,  19.62966489])

alpha *= rad
delta *= rad


#FIXED PLATE CONSTANTS 2
#alpha = np.array([ 44.45520121,  44.68462435,  45.17100625,  45.40512013,  46.74491543])
#delta = np.array([ 19.21813892,  19.27919904,  19.36062503,  19.40083402,  19.62834761])

#alpha *= rad
#delta *= rad

#FIXED PLATE CONSTANTS 3

alpha = array([ 44.45520355,  44.6846627 ,  45.17103566,  45.40516589,  46.74488949])
delta = array([ 19.21809831,  19.27913561,  19.36053285,  19.40081702,  19.62836982])

alpha *= rad
delta *= rad

#FIXED PLATE CONSTANTS 4

alpha = array([ 44.45521841,  44.68467768,  45.17098658,  45.40514157,  46.7449028 ])
delta = array([ 19.21813581,  19.27914174,  19.36052613,  19.40083071,  19.62835052])

alpha *= rad
delta *= rad

#FIXED PLATE CONSTANTS 5

alpha = array([ 0.77588993,  0.77989475,  0.78838244,  0.79246922,  0.81585246])
delta = array([ 0.33541975,  0.3364845 ,  0.33790493,  0.33860837,  0.34257934])

#ORIGINAL PLATE CONSTANT VALUES
#alpha = np.array([0.7759035722665991,0.7798931040484495,0.7883586781411437,0.7925532861101032,0.8157770734694528])
#delta = np.array([0.3356980827696327,0.33643548437860027,0.33799949331385964,0.33857448233965554,0.3427288507730832])

#JPL HORIZONS
alpha = np.array([0.7758846645330358,0.7799076484588828,0.7884001297108785,0.792482745719502,0.8149582231620589])
delta = np.array([0.3358333457866623,0.33649026832456563,0.337912711664941,0.33861132817941986,0.3425708015130415])

alpha = np.array([0.7758861189740791,0.7799091028999261,0.7884015841519219,0.7924856546015885,0.8149589503825805])
delta = np.array([0.3358333457866623,0.33649026832456563,0.33791319647862217,0.338611812993101,0.34257128632672257])

xeq = np.cos(alpha)*np.cos(delta)
yeq = np.sin(alpha)*np.cos(delta)
zeq = np.sin(delta)

epsilon = 23.43929111*rad

T = [[1,0,0],[0,np.cos(epsilon),np.sin(epsilon)],[0,-np.sin(epsilon),np.cos(epsilon)]]
T = np.matrix(T)

s = []

for i in range(len(alpha)):
	r = [xeq[i],yeq[i],zeq[i]]
	r = np.matrix(r)
	r = r.T
	ecliptic = T*r
	ecliptic = np.array(ecliptic.T)
	s.append([ecliptic[0][0],ecliptic[0][1],ecliptic[0][2]])

s = np.array(s)

earth = np.array([	[-4.852325900266916e-01,  8.563819442309909e-01, -2.676783649804444e-05],
					[-5.005639235484095e-01,  8.476786115155864e-01, -2.723171022212407e-05],
					[-5.311509164664779e-01,  8.292136310803589e-01, -2.794675828254335e-05],
					[-5.450861676152123e-01,  8.202934515047903e-01, -2.812109168370201e-05],
					[-6.143806933050102e-01,  7.707734163615326e-01, -2.700019419225799e-05]])

'''
#times = [	'2012-01-20 04:28:30',\
			'2012-01-21 04:40:27',\
			'2012-01-23 05:43:40',\
			'2012-01-24 04:26:48',\
			'2012-01-29 01:27:18']
#t = Time(times,format='iso',scale = 'utc')
'''
#Julian = t.jd
Julian = [2455946.68646, 2455947.69476, 2455949.73866, 2455950.68528, 2455955.56062]


threesome = [0,1,2,3]

s1 = np.array(s[threesome[0]])
s2 = np.array(s[threesome[1]])
s3 = np.array(s[threesome[2]])

tau1 = Julian[threesome[1]]-Julian[threesome[0]]
tau3 = Julian[threesome[2]]-Julian[threesome[1]]

s2dot = ((tau3/(tau1*(tau1+tau3)))*(s2-s1)) + ((tau1/(tau3*(tau1+tau3)))*(s3-s2))
s2double = ((2/(tau3*(tau1+tau3)))*(s3-s2)) - ((2/(tau1*(tau1+tau3)))*(s2-s1))

r0 = 2.0 #AU
radiuslist = []
rholist = []
radiuslist.append(r0)

R2 = earth[threesome[1]]

for i in range(100):
	p = rho(k,R2,s2,s2dot,s2double,radiuslist[i])
	rholist.append(p)
	rval = radius(rholist[i],R2,s2)
	radiuslist.append(rval)

rhovel = rhodot(k,R2,s2,s2dot,s2double,radiuslist[len(rholist)-1])

plt.figure()
plt.subplot(211)
plt.plot(rholist,'.')
plt.title(r'$\rho$ iterative solution')
plt.ylabel(r'$\rho$')
plt.xlabel('Iteration')
plt.subplot(212)
plt.plot(radiuslist,'.')
plt.title('radius iterative solution')
plt.ylabel('r')
plt.xlabel('Iteration')
plt.show()

asteroid = rholist[len(rholist)-1]*s2
R1 = earth[threesome[0]]
R2 = earth[threesome[1]]
R3 = earth[threesome[2]]

r = R2 + asteroid

rmag = radiuslist[len(radiuslist)-1]

R2dot = ((tau3/(tau1*(tau1+tau3)))*(R2-R1)) + ((tau1/(tau3*(tau1+tau3)))*(R3-R2))

rdot = R2dot + rholist[len(rholist)-1]*s2dot + rhovel*s2

V = np.linalg.norm(rdot)

a = (rmag*k**2)/((2*k**2)-(rmag*V**2))

h = np.cross(r,rdot)

hx,hy,hz = h

hmag = np.linalg.norm(h)

Omega = np.arctan2(-hx,hy)
Omega += np.pi
inc = np.arccos(hz/hmag)
e = np.sqrt(1-((hmag**2)/(a*k**2)))
E = np.arccos((a-rmag)/(a*e))
Eval = E
nu = trueanomaly(e,E)
M = meananomalyE(E,e)
sumnu = np.arccos((r[0]*np.cos(Omega)+r[1]*np.sin(Omega))/rmag)
omega = sumnu-nu
n = np.sqrt((k**2)/(a**3))
tau = Julian[threesome[1]]-(M/n)

period = (2*np.pi)/n

#Generate time intervals
start = tau
JulianDay = np.arange(start,start+period+500,1)
J = JulianDay - 2450000

M = meananomalytime(n,tau,start)
E0 = M

E = []
E.append(E0)

for i in range(1,len(JulianDay)):
	delE = (meananomalytime(n,tau,JulianDay[i]) - meananomalyE(E[i-1],e))/(1-e*np.cos(E[i-1]))
	En = E[i-1] + delE
	E.append(En)

E = np.array(E)
M = meananomalytime(n,tau,JulianDay)

v = trueanomaly(e,E)
r = ellipseradius(E,a,e)

theta = v+omega



#FIGURE 3

plt.figure()
plt.plot(J,E,'k',label='Eccentric Anomaly')
plt.plot(J,M,'k:',label='Mean Anomaly')
plt.xlabel('Julian Day - 2450000')
plt.ylabel('Anomaly [radians]')
plt.title('Anomaly Over Time')
plt.legend(loc='best')
plt.show()

#FIGURE 4

plt.figure()
plt.subplot(211)
plt.plot(J,r,'k')
plt.xlabel('Julian Day - 2450000')
plt.ylabel('r [AU]')
plt.title('Orbital Separation Over Time')
plt.subplot(212)
plt.plot(J,v,'k')
plt.xlabel('Julian Day - 2450000')
plt.ylabel('v [radians]')
plt.title('True Anomaly Over Time')
#plt.tight_layout()
plt.show()

#FIGURE 5

plt.figure()
plt.plot(circular(theta,a)[0],circular(theta,a)[1],':',label = 'Circular orbit radius = {0}'.format(a))
plt.plot(r*np.cos(theta),r*np.sin(theta),'k',label = 'Calculated orbit')
plt.plot(0,0,'k+')
plt.plot(r[0]*np.cos(theta[0]),r[0]*np.sin(theta[0]),'ko')
plt.axis('equal')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend(loc = 'best')
plt.title('Position of Urania in its Orbital Plane')
plt.show()

time = Julian[threesome[3]]

for i in range(len(JulianDay)):
	if np.round(JulianDay[i])==np.round(time):
		index = i

Rearth = earth[threesome[3]]

#Mnew = meananomalytime(n,tau,time)
Mnew = M[index]

#Enew = Eval + ((Mnew-M[threesome[3]])/(1-e*np.cos(Eval)))
Enew = E[index]

v = trueanomaly(e,Enew)

theta = v+omega

rmag = a*(1-e*np.cos(Enew))

rvec = [rmag*np.cos(theta),rmag*np.sin(theta),0]

TzTx = [[np.cos(Omega),-np.sin(Omega)*np.cos(inc),np.sin(Omega)*np.sin(inc)],
		[np.sin(Omega),np.cos(Omega)*np.cos(inc),-np.cos(Omega)*np.sin(inc)],
		[0,np.sin(inc),np.cos(inc)]]

TzTx = np.matrix(TzTx)

rvec = np.matrix(rvec)

r_ecliptic = TzTx*rvec.T

r_equatorial = T.T*r_ecliptic

R_earth = T.T*np.matrix(Rearth).T

R_earth = np.array(R_earth.T)

r_equatorial = np.array(r_equatorial.T)

rhos = np.array(r_ecliptic.T) - Rearth

rhos = T.T*np.matrix(rhos).T

rhos = np.array(rhos.T)

rho = np.linalg.norm(rhos)

s = rhos/rho

x,y,z = rhos[0]

alpha = np.arctan(y/x)

delta = np.arcsin(z/rho)


#ACTUAL VALUES FOR COMPARISON FROM JPL
#TAKEN ON OBS DATES AT 00:00:00


#ECCENTRICITY
EC = [1.275359615272396E-01,1.275343210138625E-01,1.275310382572875E-01,1.275293963871802E-01,1.275211899101980E-01]

#INCLINATION (degrees)
IN = [2.097732695402319E+00,2.097736408425084E+00,2.097743828715460E+00,2.097747535309888E+00,2.097766019178402E+00]

#LONGITUDE OF ASCENDING NODE (degrees)
OM = [3.076585457653067E+02,3.076584072168451E+02,3.076581353612914E+02,3.076580020201885E+02,3.076573605379849E+02]

#ARGUMENT OF PERIAPSE (degrees)
W = [8.675755976552506E+01,8.675692407772503E+01,8.675564998998844E+01,8.675501163863163E+01,8.675180796517867E+01]

#TIME OF PERIAPSE (Julian Day)
Tp = [2455833.225739588961,2455833.223487879615,2455833.218989328947,2455833.216742530465,2455833.205534818582]

#SEMI-MAJOR AXIS
A = [2.365658853240622E+00,2.365652760888053E+00,2.365640563109902E+00,2.365634458983918E+00,2.365603913071018E+00]
