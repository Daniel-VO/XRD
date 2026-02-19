"""
Created 19. Februar 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import sys
import glob
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal

def A(mu,t,tt_deg):
	return np.exp(-mu*t*(2/np.sin(np.radians(tt_deg/2))))

if len(sys.argv)>1:
	filepattern=sys.argv[1]
else:
	filepattern=''
os.system('rm '+os.path.splitext(filepattern)[0]+'*_corr.png')
os.system('rm '+os.path.splitext(filepattern)[0]+'*_bgcorr.xy')

abscorr=input('Korrektur fuer Absorption [True]? ')
if abscorr=='':
	abscorr=True
	d=eval(input('Probendicke / m: '))
else:
	abscorr=eval(abscorr)

tt,yh0=np.genfromtxt('Alu_Halter_flach_Miniflex.xy',unpack=True)
f=interpolate.interp1d(tt,yh0)

@ray.remote
def corr(i):
	fn=os.path.splitext(i)[0]
	tt_deg,yobs=np.genfromtxt(i,unpack=True)
	step=np.gradient(tt_deg)[0]

	args=(tt_deg>=min(tt))&(tt_deg<=max(tt))
	tt_deg,yobs=tt_deg[args],yobs[args]
	yh=signal.savgol_filter(f(tt_deg),int(0.5/step)+1,1)

	peaks,properties=signal.find_peaks(yh,prominence=100,width=(int(0.5/step),int(1/step)))
	argscut=[np.arange(p-2*int(properties['widths'][i]),p+2*int(properties['widths'][i])) for i,p in enumerate(peaks)]

	relints=[np.trapz(yobs[a]-min(yobs[a]),x=tt_deg[a])/np.trapz(yh[a]-min(yh[a]),x=tt_deg[a]) for a in argscut]

	params=lm.Parameters()
	params.add('mu',0,vary=abscorr)
	params.add('t',d,vary=False)
	def fitfunc(params):
		prm=params.valuesdict()
		return np.quantile(relints-A(prm['mu'],prm['t'],tt_deg[peaks]),0)
	results=lm.minimize(fitfunc,params)
	prm=results.params.valuesdict()

	# ~ plt.close('all')
	# ~ plt.plot(tt_deg[peaks],relints)
	# ~ plt.plot(tt_deg,A(prm['mu'],prm['t'],tt_deg))
	# ~ plt.show()

	Cyh=A(prm['mu'],prm['t'],180)
	argscut=np.concatenate(argscut)
	# ~ Cyh=(max(yobs[argscut])-min(yobs[argscut]))/(max(yh[argscut])-min(yh[argscut]))/2

	plt.close('all')
	plt.plot(tt_deg,yobs)
	plt.plot(tt_deg,yobs/(1-A(prm['mu'],prm['t'],tt_deg)))
	plt.plot(tt_deg,Cyh*yh)
	plt.plot(tt_deg[peaks],Cyh*yh[peaks],'+')

	yobs/=(1-A(prm['mu'],prm['t'],tt_deg))
	yobs-=Cyh*yh

	tt_deg,yobs=np.delete(tt_deg,argscut),np.delete(yobs,argscut)

	plt.plot(tt_deg,yobs)
	plt.figtext(0.15,0.15,'scale=%.3f'%Cyh+'\n mu=%.2f'%prm['mu']+'/m')

	plt.yscale('log')

	plt.xlabel('2theta/deg')
	plt.ylabel('I/1')
	plt.tight_layout(pad=0.1)
	plt.savefig(fn+'_corr.png')
	plt.savefig(fn+'_corr.pdf')

	np.savetxt(fn+'_bgcorr.xy',np.transpose([tt_deg,abs(yobs)]),fmt='%.6f')

ray.get([corr.remote(i) for i in list(filter(lambda a:'Halter' not in a,glob.glob(os.path.splitext(filepattern)[0]+'*.xy')))])
