import os
import glob
import numpy

for i in glob.glob('*[!_q].dat'):
	tt,yobs,yerr=numpy.genfromtxt(i,unpack=True)
	numpy.savetxt(os.path.splitext(i)[0]+'_q.dat',numpy.transpose([4*numpy.pi*numpy.sin(numpy.radians(tt/2))/0.15406,yobs]),fmt='%.8f')
