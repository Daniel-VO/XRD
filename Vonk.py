"""
Created 22. March 2022 by Daniel Van Opdenbosch, Technical University of Munich

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
from scipy import constants

def fsquared(vects,atoms,energy):									#Atomare Streufaktoren
	return numpy.real(numpy.average(numpy.array([i.f(2*numpy.pi*vects,en=energy) for i in atoms])**2,axis=0))

def R(vects,yobs,ycryst):											#Vonk R-Funktion
	return integrate.cumtrapz(yobs,x=vects)/integrate.cumtrapz(ycryst,x=vects)

def T(vects,atoms,energy,yobs,J):									#Vonk T-Funktion
	return integrate.cumtrapz((fsquared(vects,atoms,energy)+J)*vects**2,x=vects)/integrate.cumtrapz(yobs,x=vects)

def Vonkfunc(vects,fc,k):											#Vonk Anpassung an R
	return 1/fc+(k/(2*fc))/vects**2

def Vonksecfunc(vects,C0,C1,C2):									#Vonk Anpassung an R mit Polynom zweiten Grades
	return C0+C1*vects**2+C2*vects**4

def Vonk(filename,atoms,yobs,ycryst,twotheta_deg,emission,plots):	#Hauptfunktion Vonk.Vonk()
	L=numpy.cos(numpy.radians(twotheta_deg/2))/numpy.sin(numpy.radians(twotheta_deg))**2
	yobs/=L															#Lorentz-Korrektur anstelle von *s^2
	ycryst/=L														#Lorentz-Korrektur anstelle von *s^2
	vects=2*numpy.sin(numpy.radians(twotheta_deg/2))/xu.utilities_noconf.wavelength(emission)
	energy=xu.utilities_noconf.energy(emission)

	for i,value in enumerate(atoms):
		if isinstance(value,str):
			atoms[i]=xu.materials.atom.Atom(value[0]+value[1:].lower(),1)

	# ~ plt.plot(twotheta_deg,fsquared(vects,atoms,energy)*vects**2/(fsquared(vects,atoms,energy)*vects**2)[-1])
	# ~ plt.plot(twotheta_deg,yobs/yobs[-1])
	# ~ plt.show()

	err={}

	#Berechnung der inkohaerenten Streuung J, Korrektur von yobs
	if max(vects)>0.6:
		argsJ=numpy.where(vects[1:]>0.6)
		params=lmfit.Parameters()
		params.add('J',1,min=0)
		def VonkTfitfunc(params):
			prmT=params.valuesdict()
			return T(vects,atoms,energy,yobs,prmT['J'])[argsJ]-T(vects,atoms,energy,yobs,prmT['J'])[argsJ][-1]
		resultT=lmfit.minimize(VonkTfitfunc,params,method='least_squares')
		prmT=resultT.params.valuesdict()
		for key in resultT.params:
			err[key]=resultT.params[key].stderr
		# ~ resultT.params.pretty_print()
		yobs-=prmT['J']
		J=uq(prmT['J'],pq.dimensionless,err['J'])
	else:
		print('Warnung: Keine Bestimmung der inkohärenten Streuung - maximaler Streuvektor zu klein.')
		J=uq(0,pq.dimensionless,0)

	#Berechnung von Rulands R, Anpassung durch Vonks Funktion
	argsR=numpy.where(vects[1:]>0.6)
	params=lmfit.Parameters()
	params.add('C0',1,min=1)
	params.add('C1',0)
	params.add('C2',0)
	def VonkRfitfunc(params):
		prmR=params.valuesdict()
		return R(vects,yobs,ycryst)[argsR]-Vonksecfunc(vects,prmR['C0'],prmR['C1'],prmR['C2'])[argsR]
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

		ax1.plot(vects[argsR]**2,R(vects,yobs,ycryst)[argsR],'k',linewidth=0.5)
		ax1.plot(numpy.linspace(0,max(vects))**2,Vonksecfunc(numpy.linspace(0,max(vects)),prmR['C0'],prmR['C1'],prmR['C2']),'k--',linewidth=0.5)

		ax2.plot(vects**2,yobs,'k',linewidth=0.5)
		ax2.plot(vects**2,ycryst,'k--',linewidth=0.5)

		ax1.set_xlim([0,None])
		ax1.set_ylim([0,None])
		ax2.set_ylim([0,None])
		ax2.set_yticks([])

		ax1.set_xlabel(r'$s_p^2/\rm{\AA}^{-2}$',fontsize=10)
		ax1.set_ylabel(r'$R/1$',fontsize=10)
		ax2.set_ylabel(r'$I/1$',fontsize=10)
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
	k=2*fc*abs(uq(prmR['C1'],pq.angstrom**2,err['C1'])**2+uq(prmR['C2'],pq.angstrom**4,err['C2']))**0.5

	return fc,k,J
