import numpy as np
import PMT as pmt

nsamp = 100
tsamp = 0.001
nrep = 6

k = np.arange (nrep) +1

for i in k:
    
    x = pmt.photoncount(tsamp,nsamp)
    print x
    
    myfilename = 'dark_{0}_{1}_{2}.dat'.format(i,tsamp,nsamp)
    print 'Write data to: ' + myfilename

    np.savetxt(myfilename,x,fmt='%i')
