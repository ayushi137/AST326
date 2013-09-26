import numpy as np
import PMT as pmt

nsamp = 400
tsamp = 0.08
nrep = 6

k = np.arange (nrep) +1

for i in k:
    
    x = pmt.photoncount(tsamp,nsamp)
    print x
    
    myfilename = 'LONGCOUNTtask7pmtrunno_{0}count{1}.dat'.format(i,tsamp)
    print 'Write data to: ' + myfilename

    np.savetxt(myfilename,x,fmt='%i')
