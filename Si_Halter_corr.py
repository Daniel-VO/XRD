"""
Created 26. September 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import numpy
import scipy
import lmfit
import matplotlib.pyplot as plt
import xrayutilities as xu

def fsquared(vects,atoms,energy):												#Atomare Streufaktoren
	return numpy.real(numpy.average(numpy.array([i.f(2*numpy.pi*vects,en=energy) for i in atoms])**2,axis=0))

if len(sys.argv)>1:
	filepattern=sys.argv[1]
else:
	filepattern=''
os.system('rm '+os.path.splitext(filepattern)[0]+'*_corr.png')
os.system('rm '+os.path.splitext(filepattern)[0]+'*_bgcorr.xy')
files=list(filter(lambda a:'Halter' not in a,glob.glob(os.path.splitext(filepattern)[0]+'*.xy')))

tt,yh0=numpy.genfromtxt('Si_Halter.xy',unpack=True)
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








	yh,emission=scipy.signal.savgol_filter(yh,101,1),'CuKa1'
	vects=2*numpy.sin(numpy.radians(twotheta_deg/2))/xu.utilities_noconf.wavelength(emission)
	Lfsq=	1/(numpy.sin(numpy.radians(twotheta_deg)))*\
			fsquared(vects,\
			1*[xu.materials.atom.Atom('C',1)]+\
			4*[xu.materials.atom.Atom('H',1)],xu.utilities_noconf.energy(emission))
	params=lmfit.Parameters()
	params.add('Cyh',1)
	def minfunc(params):
		prm=params.valuesdict()
		return scipy.integrate.cumtrapz(Lfsq*vects**2,x=vects)/numpy.trapz(Lfsq*vects**2,x=vects)-scipy.integrate.cumtrapz((yobs-prm['Cyh']*yh)*vects**2,x=vects)/numpy.trapz((yobs-prm['Cyh']*yh)*vects**2,x=vects)
	result=lmfit.minimize(minfunc,params)
	prm=result.params.valuesdict()
	yobs-=prm['Cyh']*yh

	plt.plot(twotheta_deg,yobs)
	plt.plot(twotheta_deg,Lfsq/Lfsq[-1]*yobs[-1])
	plt.plot(twotheta_deg,prm['Cyh']*yh)
	plt.text(min(twotheta_deg),min(prm['Cyh']*yh),prm['Cyh'].round(3))

	plt.yscale('log')
	plt.savefig(filename+'_corr.png')

	numpy.savetxt(filename+'_bgcorr.xy',numpy.transpose([twotheta_deg,abs(yobs)]))
