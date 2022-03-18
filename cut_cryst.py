"""
Created 18. March 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import glob
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

files=glob.glob('*[!_cut].xy')

Fenster=0.3
fract=12
anglelim=30

@ray.remote
def cut(i):
	zwotheta,yobs=numpy.genfromtxt(i,unpack=True)
	yobs_sm=signal.savgol_filter(yobs,int(Fenster/numpy.diff(zwotheta)[0]+1),1)
	args=numpy.where((abs(numpy.gradient(yobs_sm))>max(abs(numpy.gradient(yobs_sm)))/fract/numpy.sin(numpy.radians(zwotheta)))&(zwotheta>anglelim))[0]
	for a in args:
		args=numpy.append(args,numpy.arange(a-Fenster/numpy.diff(zwotheta)[0],a+Fenster/numpy.diff(zwotheta)[0],dtype=int))
	# ~ print(i)
	# ~ plt.plot(zwotheta,yobs)
	# ~ plt.scatter(zwotheta[args],yobs[args],c='r')
	# ~ plt.show()
	numpy.savetxt(os.path.splitext(i)[0]+'_cut'+os.path.splitext(i)[1],numpy.transpose([numpy.delete(zwotheta,args[numpy.where(args<len(zwotheta))]),numpy.delete(yobs,args[numpy.where(args<len(zwotheta))])]))

ray.get([cut.remote(i) for i in files])
