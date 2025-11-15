"""
Created 14. November 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal

if len(sys.argv)>1:
	filepattern=sys.argv[1]
else:
	filepattern=''
os.system('rm '+os.path.splitext(filepattern)[0]+'*_corr.png')
os.system('rm '+os.path.splitext(filepattern)[0]+'*_bgcorr.xy')

tt,yh0=np.genfromtxt('Alu_Halter_flach_Miniflex.xy',unpack=True)
f=interpolate.interp1d(tt,yh0)

for i in list(filter(lambda a:'Halter' not in a,glob.glob(os.path.splitext(filepattern)[0]+'*.xy'))):
	filename=os.path.splitext(i)[0]
	tt_deg,yobs=np.genfromtxt(i,unpack=True)
	step=np.gradient(tt_deg)[0]

	args=np.where((tt_deg>=min(tt))&(tt_deg<=max(tt)))
	tt_deg,yobs=tt_deg[args],yobs[args]
	yh=signal.savgol_filter(f(tt_deg),int(0.5/step)+1,1)

	peaks,properties=signal.find_peaks(yh,prominence=100,width=(int(0.5/step),int(1/step)))
	argscut=[np.arange(p-2*int(properties['widths'][i]),p+2*int(properties['widths'][i])) for i,p in enumerate(peaks)]

	Cyh=np.average([np.trapz(yobs[a]-min(yobs[a]),x=tt_deg[a])/np.trapz(yh[a]-min(yh[a]),x=tt_deg[a]) for a in argscut])
	argscut=np.concatenate(argscut)
	# ~ Cyh=(max(yobs[argscut])-min(yobs[argscut]))/(max(yh[argscut])-min(yh[argscut]))/2

	plt.close('all')
	plt.plot(tt_deg,yobs)
	plt.plot(tt_deg,Cyh*yh)
	plt.plot(tt_deg[peaks],yh[peaks],'+')

	yobs-=Cyh*yh

	tt_deg,yobs=np.delete(tt_deg,argscut),np.delete(yobs,argscut)

	plt.plot(tt_deg,yobs)
	plt.text(min(tt_deg),min(Cyh*yh),Cyh.round(3))

	plt.yscale('log')
	plt.savefig(filename+'_corr.png')
	plt.savefig(filename+'_corr.pdf')

	np.savetxt(filename+'_bgcorr.xy',np.transpose([tt_deg,yobs]),fmt='%.6f')
