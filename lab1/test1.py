import numpy as np
import PMT as pmt

nsamp = 100
tsamp = 0.01
nrep = 4

k = np.arange (nrep) +1

for i in k:
    
    x = pmt.photoncount(tsamp,nsamp)
    print x
    
    myfilename = 'test_{:02d}_{:0.3f}_{:04d}.dat'.format(i,tsamp,nsamp)
    print 'Write data to: ' + myfilename

    np.savetxt(myfilename,x,fmt='%i')
