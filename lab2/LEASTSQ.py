#least squaresfit
import numpy as np
from matplotlib import pylab as plt

#Pixels = np.array([1223.5, 1242.5, 1281.5, 1364.5, 1378.5, 1408.5, 1422.5, 1489.5, 1535.5, 1567.5, 1580.5, 1652.5])
#Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59,6266.49,6334.43,6382.99,6402.25,6506.53])

#Neon2
Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59, 6217.28, 6266.49,6334.43,6382.99,6402.25,6506.53, 6532.88, 6598.95,6678.28, 6717.04, 6929.47,7032.41, 7173.94, 7245.17, 7438.90 ])
#Neon1
#Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59, 6217.28, 6266.49,6304.79, 6334.43,6382.99,6402.25,6506.53, 6532.88, 6598.95,6678.28, 6717.04, 6929.47,7032.41 ])

Pixels = np.loadtxt("Telescope_Neon2.txt", usecols = (0,))


ma =np.array([[np.sum(Pixels**2),np.sum(Pixels)],[np.sum(Pixels),len(Pixels)]])
mc =np.array([[np.sum(Pixels*Wavelengths)],[np.sum(Wavelengths)]])

mai = np.linalg.inv(ma)
md = np.dot(mai,mc)

mfit = md[0,0]
cfit = md[1,0]

variance = (1.0/(len(Pixels)-2))*np.sum(Wavelengths-Pixels*mfit-cfit)**2.0
residual = Wavelengths - mfit*Pixels+cfit

print mfit,cfit,variance


plt.figure(1)
plt.plot(Pixels,Wavelengths,'o',label="data")
plt.plot(Pixels,mfit*Pixels+cfit)
plt.text(275, 7300, 'c = 4846.31')

plt.text(275, 7200, 'm = 3.848')
plt.text(275, 7100, 'variance = 3.95*10^(-21)')
plt.xlabel("Pixel Number")
plt.ylabel("Wavelength (Angstrom)")
plt.title("Line of Best Fit for Neon (Night 2)")

plt.figure(2)
plt.scatter (Pixels, residual)
plt.xlabel("Pixel Number")
plt.ylabel("Residual")
plt.title("Linear Residual (Night 2)")


plt.show()