
import pyfits as pf
import matplotlib.pyplot as plt
import numpy as np


#data = "Night1/Neon_30sec_01.fit"
#data2 = "Night1/Dark_30sec_01.fit"
data = "Night1/Lamp1.fit"
data2 = "Night1/Lamp1_dark.fit"

line = pf.open(data)
darl = pf.open(data2)

# get the header
head_line = pf.getheader(data)

# get the data 
lin = line[0].data
dark = darl[0].data
intensity = lin[300]
#np.savetxt("Lamp1.txt", intensity, fmt='%.2f')
dark_intensity = dark[300]
pixel = np.arange(0, len(intensity),1)
fixed_intensity = intensity #- dark_intensity
fixed_intensity  = fixed_intensity[::-1]
dark_intensity = dark_intensity[::-1]

'''
plt.figure(1)
plt.plot(pixel, intensity)
plt.plot(pixel, dark_intensity)
plt.plot(pixel, fixed_intensity)
'''

centroidpixel = np.zeros(1)
centroidintensity = np.zeros(1)
for i in range(len(fixed_intensity)-2):
    if fixed_intensity[i+1]>fixed_intensity[i]:
        i += 1
        if fixed_intensity[i+1]<=fixed_intensity[i]:
            if fixed_intensity[i]>180:
                #cpixel = 0.5*(pixel[i]+pixel[i+1])
                centroidpixel = np.append(centroidpixel,pixel[i])
                    
                #cintensity = 0.5*(intensity[i]+intensity[i+1])
                centroidintensity= np.append(centroidintensity, fixed_intensity[i])
centroidpixel = np.delete(centroidpixel, 0)
centroidintensity = np.delete(centroidintensity, 0)
output = np.column_stack((centroidpixel, centroidintensity))

plt.figure(1)
plt.plot(pixel, fixed_intensity)
plt.plot(503,fixed_intensity[503], 'or' )
plt.title ("Neon from Night1")
plt.xlabel ("Pixels")
plt.ylabel ("Intensity")

#np.savetxt("Telescope_Neon.txt", output, fmt='%.2f')

plt.figure(2)
plt.plot(pixel, dark_intensity)
'''
plt.figure(3)
plt.imshow(dark, origin = 'lower', interpolation ='nearest')
plt.colorbar()
'''
plt.show()
