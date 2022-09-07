"""
Created 07. September 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy
import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import xrayutilities as xu
import quantities as pq
from quantities import UncertainQuantity as uq
from scipy import integrate

def fsquared(vects,atoms,energy):												#Atomare Streufaktoren
	return numpy.real(numpy.average(numpy.array([i.f(2*numpy.pi*vects,en=energy) for i in atoms])**2,axis=0))

def R(vects,yobs,ycryst):														#Vonk R-Funktion
	return integrate.cumtrapz(yobs,x=vects)/integrate.cumtrapz(ycryst,x=vects)

def T(vects,atoms,energy,yobs,J):												#Vonk T-Funktion
	return integrate.cumtrapz(fsquared(vects,atoms,energy)*vects**2,x=vects)/integrate.cumtrapz(yobs-J*vects**2,x=vects)

def Vonkfunc(vects,fc,k):														#Vonk Anpassung an R
	return 1/fc+(k/(2*fc))/vects**2

def polysecond(x,C0,C1,C2):														#Anpassung an R mit Polynom zweiten Grades
	return C0+C1*x+C2*x**2

def Vonk(filename,atoms,yobs,ycryst,twotheta_deg,emission,plots,lowerbound,incohcor):	#Hauptfunktion Vonk.Vonk()
	L=1/(2*numpy.sin(numpy.radians(twotheta_deg/2))*numpy.sin(numpy.radians(twotheta_deg)))
	yobs/=L																		#Lorentz-Korrektur anstelle von *s^2
	ycryst/=L																	#Lorentz-Korrektur anstelle von *s^2
	vects=2*numpy.sin(numpy.radians(twotheta_deg/2))/xu.utilities_noconf.wavelength(emission)
	energy=xu.utilities_noconf.energy(emission)

	for i,value in enumerate(atoms):
		if isinstance(value,str):
			atoms[i]=xu.materials.atom.Atom(value[0]+value[1:].lower(),1)

	err={}
	args=numpy.where(vects[1:]>lowerbound)

	#Berechnung der inkohaerenten Streuung J, Korrektur von yobs
	if incohcor==True:
		params=lmfit.Parameters()
		params.add('J',1,min=0)
		def VonkTfitfunc(params):
			prmT=params.valuesdict()
			return T(vects,atoms,energy,yobs,prmT['J'])[args]-T(vects,atoms,energy,yobs,prmT['J'])[args][-1]
		resultT=lmfit.minimize(VonkTfitfunc,params,method='least_squares')
		prmT=resultT.params.valuesdict()
		for key in resultT.params:
			err[key]=resultT.params[key].stderr
		# ~ resultT.params.pretty_print()
		yobs-=prmT['J']*vects**2
		J=uq(prmT['J'],pq.dimensionless,err['J'])
	else:
		J=uq(0,pq.dimensionless,0)

	#Normierung auf elektronische Einheiten eA^-2
	normfakt=numpy.median((fsquared(vects,atoms,energy)*vects**2/yobs)[numpy.where(twotheta_deg>max(twotheta_deg)-2)])
	yobs*=normfakt
	ycryst*=normfakt

	#Berechnung von Rulands R, Anpassung durch Vonks Funktion
	RulandR=R(vects,yobs,ycryst)
	params=lmfit.Parameters()
	params.add('C0',1,min=1)
	params.add('C1',0,min=0)
	params.add('C2',0)
	def VonkRfitfunc(params):
		prmR=params.valuesdict()
		return RulandR[args]-polysecond(vects**2,prmR['C0'],prmR['C1'],prmR['C2'])[args]
	resultR=lmfit.minimize(VonkRfitfunc,params,method='least_squares')
	prmR=resultR.params.valuesdict()
	for key in resultR.params:
		err[key]=resultR.params[key].stderr
	# ~ resultR.params.pretty_print()

	#Abbildungen
	if plots==True:
		plt.clf()
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		fig,ax1=plt.subplots(figsize=(7.5/2.54,5.3/2.54))
		ax2=ax1.twinx()

		ax1.plot(vects[args]**2,RulandR[args],'k',linewidth=0.5)
		ax1.plot(numpy.linspace(0,max(vects))**2,polysecond(numpy.linspace(0,max(vects))**2,prmR['C0'],prmR['C1'],prmR['C2']),'k--',linewidth=0.5)

		ax2.plot(vects**2,yobs,'k',linewidth=0.5)
		ax2.plot(vects**2,ycryst,'k--',linewidth=0.5)
		ax2.plot(vects**2,fsquared(vects,atoms,energy)*vects**2,'w',linewidth=0.5)
		ax2.plot(vects**2,fsquared(vects,atoms,energy)*vects**2,'k:',linewidth=0.5)

		ax1.set_xlim([0,None])
		ax1.set_ylim([0,None])
		ax2.set_ylim([0,None])

		ax1.set_xlabel(r'$s_p^2/\rm{\AA}^{-2}$',fontsize=10)
		ax1.set_ylabel(r'$R/1$',fontsize=10)
		ax2.set_ylabel(r'$Is^2/(\rm{e\,\AA}^{-2})$',fontsize=10)
		ax1.tick_params(direction='out')
		ax1.tick_params(axis='x',pad=2,labelsize=8)
		ax1.tick_params(axis='y',pad=2,labelsize=8)
		ax2.tick_params(axis='y',pad=2,labelsize=8)
		ax1.xaxis.get_offset_text().set_size(8)
		ax1.yaxis.get_offset_text().set_size(8)
		ax2.yaxis.get_offset_text().set_size(8)
		plt.tight_layout(pad=0.1)
		plt.savefig(filename+'_Vonk.png',dpi=300)
		plt.close('all')

	fc=1/uq(prmR['C0'],pq.dimensionless,err['C0'])
	k=2*fc*uq(prmR['C1'],pq.angstrom**2,err['C1'])

	return fc,k,J
