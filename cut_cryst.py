import os
import glob
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

files=glob.glob('*[!_cut].xy')

Fenster=0.5

for i in files:
	zwotheta,yobs=numpy.genfromtxt(i,unpack=True)
	args=numpy.where(abs(numpy.gradient(yobs))>max(abs(numpy.gradient(yobs)))/30/numpy.sin(numpy.radians(zwotheta)))[0]
	for a in args:
		args=numpy.append(args,numpy.arange(a-Fenster/numpy.diff(zwotheta)[0],a+Fenster/numpy.diff(zwotheta)[0],dtype=int))
	# ~ print(i)
	# ~ plt.plot(zwotheta,yobs)
	# ~ plt.scatter(zwotheta[args],yobs[args],c='r')
	# ~ plt.show()
	numpy.savetxt(os.path.splitext(i)[0]+'_cut'+os.path.splitext(i)[1],numpy.transpose([numpy.delete(zwotheta,args),numpy.delete(yobs,args)]))
