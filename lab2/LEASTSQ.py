#least squaresfit
import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import leastsq
import matplotlib.gridspec as gridspec

#Pixels = np.array([1223.5, 1242.5, 1281.5, 1364.5, 1378.5, 1408.5, 1422.5, 1489.5, 1535.5, 1567.5, 1580.5, 1652.5])
# Neon CCD
Pixels = np.loadtxt("Telescope_Neon.txt", usecols = (0,))
#Pixels = np.loadtxt("centroid_for_CCD_Neon1000.txt", usecols = (0,))
#Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59, 6217.28, 6266.49, 6334.43,6382.99,6402.25,6506.53, 6532.88, 6598.95,6678.28, 6717.04, 6929.47,7032.41 ])



#Neon2
#Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59, 6217.28, 6266.49,6334.43,6382.99,6402.25,6506.53, 6532.88, 6598.95,6678.28, 6717.04, 6929.47,7032.41, 7173.94, 7245.17, 7438.90 ])
#Neon1
Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59, 6217.28, 6266.49,6304.79, 6334.43,6382.99,6402.25,6506.53, 6532.88, 6598.95,6678.28, 6717.04, 6929.47,7032.41 ])




ma =np.array([[np.sum(Pixels**2),np.sum(Pixels)],[np.sum(Pixels),len(Pixels)]])
mc =np.array([[np.sum(Pixels*Wavelengths)],[np.sum(Wavelengths)]])

mai = np.linalg.inv(ma)
md = np.dot(mai,mc)

mfit = md[0,0]
cfit = md[1,0]
#mfit = 2.033
#cfit = 3628.99

variance = (1.0/(len(Pixels)-2))*np.sum(Wavelengths-Pixels*mfit-cfit)**2.0
residual = Wavelengths - (mfit*Pixels+cfit)

print mfit,cfit,variance


############## quadratic #################################

x = Pixels
y = Wavelengths

#dy = 0.5
p0 = [2.012, 3645.85, -0.00017]
#fit function
def peval(x, p):
    return (p[1]+(p[0]*x)+(p[2]*(x**2)))

def residuals (p,y,x, peval):
    return (y) - peval(x,p)

p_final = leastsq(residuals,p0,args=(y,x, peval), full_output= True,maxfev=2000)


y_final = peval(x,p_final[0])
chi2 = np.sum((y - y_final)**2)#/ ((dy)**2))
resi = (residuals(p_final[0],y,x,peval))
dof = len(y)-len(p0)
chi_re2 = chi2/dof # residual variance
cov = p_final[1] * chi_re2
cov_xy = cov[0][1]
cov_x = np.sqrt(cov[0][0])
cov_y = np.sqrt(cov[1][1])
r =cov_xy/(cov_x*cov_y)

print "The inital parameter (p0) we used is:\n", p0
print "What we get as a parameter:", p_final[0]

if p_final[4] == 1: # change p_final[1] to success
    print "It converges."
else:
    print "It does not converge."

print "The Chi square is: \t\t\t",round(chi2,2)
print "The Chi-reduced square is: \t\t", round(chi_re2,2)
print
print "Cov_xy:",round(cov_xy,4), "\nCov_x: ",round(cov_x,4),"\nCov_y: ", round(cov_y,4)
print "Sample coefficient of linear correlation (r): ", round(r,2)
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print


'''
plt.figure(1)
plt.subplot(2,1,1)
plt.scatter(x,y)
plt.plot(x,peval(x, p_final[0]), color='r',label='fit plot')

'''

plt.figure(1 , figsize=(9,6))
plt.subplot(2,1,1)
plt.plot(Pixels,Wavelengths,'o',label="data")
plt.plot(Pixels,peval(Pixels, p_final[0]), color='r',label='fit plot')
plt.legend(loc=4)

'''
# Night2
plt.text(275, 7300, 'a_1 = 4846.31')
plt.text(275, 7200, 'a_2 = 3.848')
plt.text(275, 7100, 'a_2 = -1.76E(-4))')
plt.text(275, 7000, 'variance = 3.95E(-21)')
'''

#Night 1
plt.text(475, 7000, 'a_0 = 3828.52')
plt.text(475, 6900, 'a_1 = 4.21')
#plt.text(475, 6800, 'a_2 = 4.72E(-5))')
plt.text(475, 6800, 'variance = 3.80E(-20)')

#wavelength = 3828.52 + (4.21*pixel)
'''
#CCD
plt.text(1230, 7000, 'a_0 = 3645.85')
plt.text(1230, 6900, 'a_1 = 2.01')
plt.text(1230, 6800, 'a_2 = -1.76E(-4))')
plt.text(1230, 6700, 'variance = 5.23E(-21)')
'''
#wavelength = 3645.85 + (2.01*pixel) +(-0.000176*(pixel**2))

plt.xlabel("Pixel Number")
plt.ylabel("Wavelength (Angstrom)")
plt.title("Wavelength Fitting for Neon")

#plt.figure(2, figsize=(9,3))

plt.subplot(2,1,2)
plt.scatter (Pixels, residual)
plt.xlabel("Pixel Number")
plt.ylabel("Residual")
plt.title("Linear Residual")
'''
plt.subplot(1,2,2)
plt.scatter (Pixels, resi)
plt.xlabel("Pixel Number")
plt.ylabel("Residual")
plt.title("Quadratic Residual")
'''
plt.tight_layout()


plt.show()