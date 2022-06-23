"""
Created 23. June 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import numpy
import matplotlib.pyplot as plt

from scipy import interpolate


os.system('rm '+os.path.splitext(sys.argv[1])[0]+'*_corr.png')
os.system('rm '+os.path.splitext(sys.argv[1])[0]+'*_bgcorr.xy')
files=list(filter(lambda a:'Halter' not in a,glob.glob(os.path.splitext(sys.argv[1])[0]+'*.xy')))

tt,yh0=numpy.genfromtxt('Alu_Halter_flach.xy',unpack=True)
f=interpolate.interp1d(tt,yh0)

for i in files:
	filename=os.path.splitext(i)[0]
	twotheta_deg,yobs=numpy.genfromtxt(i,unpack=True)

	args=numpy.where(twotheta_deg<=max(tt))
	twotheta_deg,yobs=twotheta_deg[args],yobs[args]
	yh=f(twotheta_deg)

	plt.clf()
	plt.plot(twotheta_deg,yobs)
	plt.plot(twotheta_deg,yh)

	ttmax=twotheta_deg[numpy.where(yh==max(yh))]
	argssubt=numpy.where((twotheta_deg>ttmax-2)&(twotheta_deg<ttmax+2))
	yobs-=yh*(max(yobs[argssubt])-min(yobs[argssubt]))/(max(yh[argssubt])-min(yh[argssubt]))

	step,argscut=numpy.diff(twotheta_deg)[0],[]
	for i,valuei in enumerate(yh):
		if twotheta_deg[i]>12 and valuei>min(yh[i-int(2/step):i+int(2/step)])*2:
			argscut.append(numpy.arange(i-int(1/2/step),i+int(1/2/step)))
	twotheta_deg,yobs=numpy.delete(twotheta_deg,argscut),numpy.delete(yobs,argscut)

	plt.plot(twotheta_deg,yobs)


	plt.yscale('log')
	plt.savefig(filename+'_corr.png')

	numpy.savetxt(filename+'_bgcorr.xy',numpy.transpose([twotheta_deg,abs(yobs)]))
