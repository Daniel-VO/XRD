"""
Created 26. September 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import numpy
import scipy
import matplotlib.pyplot as plt

if len(sys.argv)>1:
	filepattern=sys.argv[1]
else:
	filepattern=''
os.system('rm '+os.path.splitext(filepattern)[0]+'*_corr.png')
os.system('rm '+os.path.splitext(filepattern)[0]+'*_bgcorr.xy')
files=list(filter(lambda a:'Halter' not in a,glob.glob(os.path.splitext(filepattern)[0]+'*.xy')))

tt,yh0=numpy.genfromtxt('Alu_Halter_flach.xy',unpack=True)
f=scipy.interpolate.interp1d(tt,yh0)

for i in files:
	filename=os.path.splitext(i)[0]
	twotheta_deg,yobs=numpy.genfromtxt(i,unpack=True)

	args=numpy.where(twotheta_deg<=max(tt))
	twotheta_deg,yobs=twotheta_deg[args],yobs[args]
	yh=f(twotheta_deg)

	plt.clf()
	plt.plot(twotheta_deg,yobs)
	plt.plot(twotheta_deg,yh)

	step,argscut=numpy.gradient(twotheta_deg)[0],[]
	for i,valuei in enumerate(yh):
		if twotheta_deg[i]>min(twotheta_deg)+2 and valuei>min(yh[i-int(2/step):i+int(2/step)])*2:
			argscut.append(numpy.arange(i-int(1/step),i+int(1/step)))
	Cyh=numpy.trapz(yobs[argscut].flatten(),x=twotheta_deg[argscut].flatten())/numpy.trapz(yh[argscut].flatten(),x=twotheta_deg[argscut].flatten())
	twotheta_deg,yobs,yh=numpy.delete(twotheta_deg,argscut),numpy.delete(yobs,argscut),numpy.delete(yh,argscut)

	yh=scipy.signal.savgol_filter(yh,101,1)
	yobs-=Cyh*yh

	plt.plot(twotheta_deg,yobs)
	plt.plot(twotheta_deg,Cyh*yh)
	plt.text(min(twotheta_deg),min(Cyh*yh),Cyh.round(3))

	plt.yscale('log')
	plt.savefig(filename+'_corr.png')

	numpy.savetxt(filename+'_bgcorr.xy',numpy.transpose([twotheta_deg,abs(yobs)]))
