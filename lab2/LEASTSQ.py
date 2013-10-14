#least squaresfit
import numpy as np
from matplotlib import pylab as plt

Pixels = np.array([1223.5, 1242.5, 1281.5, 1364.5, 1378.5, 1408.5, 1422.5, 1489.5, 1535.5, 1567.5, 1580.5, 1652.5])
Wavelengths = np.array([5852.49,5881.89,5944.83,6074.34,6096.16,6143.06,6163.59,6266.49,6334.43,6382.99,6402.25,6506.53])

ma =np.array([[np.sum(Pixels**2),np.sum(Pixels)],[np.sum(Pixels),len(Pixels)]])
mc =np.array([[np.sum(Pixels*Wavelengths)],[np.sum(Wavelengths)]])

mai = np.linalg.inv(ma)
md = np.dot(mai,mc)

mfit = md[0,0]
cfit = md[1,0]

plt.plot(Pixels,Wavelengths,'o',label="data")
plt.plot(Pixels,mfit*Pixels+cfit)
plt.xlabel("Pixel Number")
plt.ylabel("Wavelength (Angstroms)")
plt.title("Line of Best Fit")
plt.show()

sigmasquared = (1.0/(len(Pixels)-2))*np.sum(Wavelengths-Pixels*mfit-cfit)**2.0

print mfit,cfit,sigmasquared
