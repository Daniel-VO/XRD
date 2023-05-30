"""
Created 30. May 2023 by Daniel Van Opdenbosch, Technical University of Munich

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

tt,yh0=np.genfromtxt('Stahl_Halter_DCS500.xy',unpack=True)
f=interpolate.interp1d(tt,yh0)

for i in list(filter(lambda a:'Halter' not in a,glob.glob(os.path.splitext(filepattern)[0]+'*.xy'))):
	filename=os.path.splitext(i)[0]
	twotheta_deg,yobs=np.genfromtxt(i,unpack=True)

	args=np.where((twotheta_deg>=min(tt))&(twotheta_deg<=max(tt)))
	twotheta_deg,yobs=twotheta_deg[args],yobs[args]
	yh=signal.savgol_filter(f(twotheta_deg),51,1)

	step,argscut=np.gradient(twotheta_deg)[0],[]
	for i,valuei in enumerate(yh):
		if twotheta_deg[i]>min(twotheta_deg)+2 and valuei>min(yh[i-int(2/step):i+int(2/step)])*2:
			argscut.append(np.arange(i-int(1/2/step),i+int(1/2/step)))
	Cyh=np.trapz(yobs[argscut].flatten(),x=twotheta_deg[argscut].flatten())/np.trapz(yh[argscut].flatten(),x=twotheta_deg[argscut].flatten())

	plt.close('all')
	plt.plot(twotheta_deg,yobs)
	plt.plot(twotheta_deg,Cyh*yh)

	plt.plot(twotheta_deg,Cyh*yh)

	yobs-=Cyh*yh

	twotheta_deg,yobs=np.delete(twotheta_deg,argscut),np.delete(yobs,argscut)

	plt.plot(twotheta_deg,yobs)
	plt.text(min(twotheta_deg),min(Cyh*yh),Cyh.round(3))

	plt.yscale('log')
	plt.savefig(filename+'_corr.png')

	np.savetxt(filename+'_bgcorr.xy',np.transpose([twotheta_deg,abs(yobs)]),fmt='%.6f')
