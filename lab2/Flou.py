import numpy as np
import matplotlib.pyplot as plt

#first find centroids
d = []
trial2 = np.arange(1,11,1)
for n in trial2:
    k = ""
    if n < 10:
        k = "0000"
    elif n >= 10 and n < 100:
        k = "000"
    elif n >= 100 and n <1000:
        k = "00"
    else:
        k = "0"
    dark = np.loadtxt("Dark/100ms_Dark{0}{1}.txt".format(k,n),usecols=(1,))
    d.append(dark)


sourcelist = ["FLUOR"]
p = []
inten = []

for source in sourcelist:
    trial = np.arange(1,7,1)
    for n in trial:
        
        k = ""
        if n < 10:
            k = "0000"
        elif n >= 10 and n < 100:
            k = "000"
        elif n >= 100 and n <1000:
            k = "00"
        else:
            k = "0"
        
        pixel = np.loadtxt("Old_Data/100ms{0}{1}.txt".format(source,n),usecols=(0,))
        p.append(pixel)
        intensity = np.loadtxt("Old_Data/100ms{0}{1}.txt".format(source,n),usecols=(1,))
        inten.append(intensity)

p = np.array(p)
d = np.array(d)
inten = np.array(inten)

pix = np.transpose(p)
d = np.transpose(d)
inte = np.transpose(inten)
pixel = []
intensity = []
dark = []
for i in range(len(pix)):
    pixel.append(np.average(pix[i]))
    dark.append(np.average(d[i]))
    intensity.append(np.average(inte[i]) - np.average(d[i]))


centroidpixel = np.zeros(1)
centroidintensity = np.zeros(1)
for i in range(len(intensity)-2):
    if intensity[i+1]>intensity[i]:
        i += 1
        if intensity[i+1]<=intensity[i]:
            c = 5000#intensity[1000]+4000 # change this value to get even smaller peak
            if intensity[i]>c:
                #cpixel = 0.5*(pixel[i]+pixel[i+1])
                centroidpixel = np.append(centroidpixel,pixel[i])
                        
                #cintensity = 0.5*(intensity[i]+intensity[i+1])
                centroidintensity= np.append(centroidintensity, intensity[i])
centroidpixel = np.delete(centroidpixel, 0)
centroidintensity = np.delete(centroidintensity, 0)
output = np.column_stack((centroidpixel, centroidintensity))
centroidintensity = np.array(centroidintensity)
centroidpixel = np.array(centroidpixel)
pixel = np.array(pixel)
intensity = np.array(intensity)
centroidpixel = np.loadtxt("centroid_for_CCD_FBulb.txt",usecols=(0,))
centroidintensity = np.loadtxt("centroid_for_CCD_FBulb.txt",usecols=(1,))


wavelength = 3645.85 + (2.01*pixel) +(-0.000176*(pixel**2)) # CCD
wavelength2 = 3645.85 + (2.01*centroidpixel) +(-0.000176*(centroidpixel**2)) # CCD

######################### plots ###################################
plt.figure()
plt.plot(wavelength, intensity, color='b')
plt.plot(wavelength2,centroidintensity, 'or' )
plt.title ("Averaged Fluorescent Bulb Spectrum with denoted peaks")
#plt.xlim(0,2070)
plt.xlim(3600,7050)
plt.ylim(-100)
plt.xlabel("Wavelength")
plt.ylabel("Intensity")
plt.text (4000, 11500, 'Hg')
plt.text (4300, 48000, 'Hg')
plt.text (4800, 6300, 'Tb3+')
plt.text (5200, 41000, 'Tb3+')
plt.text (5400, 66000, 'Hg')
plt.text (5700, 25000, 'Hg, Tb3+')
plt.text (5700, 22000, 'or Eu+3')
plt.text (6000, 63500, 'Eu+3')


###################### saving files ##############################
#np.savetxt("centroid_for_CCD_FBulb.txt".format(source,n), output, fmt='%.2f')

plt.show()
