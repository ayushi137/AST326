import numpy as np
import matplotlib.pyplot as plt

sourcelist = ["1s", "200ms", "100ms","50ms", ]


Pixels = []
Intensity = []
for source in sourcelist:
    p = []
    inten = []
    d = []
    trial = np.arange(1,11,1)
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
        
        pixel = np.loadtxt("Lamp/{0}_Lamp{1}{2}.txt".format(source,k,n),usecols=(0,))
        p.append(pixel)
        intensity = np.loadtxt("Lamp/{0}_Lamp{1}{2}.txt".format(source,k,n),usecols=(1,))
        inten.append(intensity)
        dark = np.loadtxt("Dark/{0}_Dark{1}{2}.txt".format(source,k,n),usecols=(1,))
        d.append(dark)

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
        intensity.append((np.average(inte[i]) - np.average(d[i]))/1.37)

    Pixels.append(pixel)
    Intensity.append(intensity)

plt.figure()
k =0
for i in Intensity:
    plt.plot(Pixels[0],i, label="{0}".format(sourcelist[k]))
    k+=1
plt.legend()
plt.ylim(0)
plt.xlim(0,2700)
plt.title("Lamp Spectra at different Exposure time")
plt.xlabel("Pixels")
plt.ylabel("Intensity")

plt.show()