"""
Created 23. October 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import scipy
import numpy as np
import lmfit as lm
import matplotlib as mpl
import matplotlib.pyplot as plt
import xrayutilities as xu
import quantities as pq
from quantities import UncertainQuantity as uq

def fsquared(vects,atoms,energy):												#Atomare Streufaktoren
	return np.real(np.average(np.array([i.f(2*np.pi*vects,en=energy) for i in atoms])**2,axis=0))

def R(vects,yobs,ycoh):														#Vonk R-Funktion
	return scipy.integrate.cumulative_trapezoid(yobs,x=vects)/scipy.integrate.cumulative_trapezoid(ycoh,x=vects)

def T(vects,atoms,energy,yobs,J):												#Vonk T-Funktion
	return scipy.integrate.cumulative_trapezoid(fsquared(vects,atoms,energy)*vects**2,x=vects)/scipy.integrate.cumulative_trapezoid(yobs-J*vects**2,x=vects)

def Vonkfunc(vects,xc,k):														#Vonk Anpassung an R
	return 1/xc+(k/(2*xc))/vects**2

def polysecond(x,C0,C1,C2):														#Anpassung an R mit Polynom zweiten Grades
	return C0+C1*x+C2*x**2

def Vonk(filename,atoms,yobs,ycoh,twotheta_deg,emission,inelcor,varslitcor):	#Hauptfunktion Vonk.Vonk()
	if varslitcor:
		varslitcor=np.sin(np.radians(twotheta_deg/2))
		yobs/=varslitcor;ycoh/=varslitcor
	vects=2*np.sin(np.radians(twotheta_deg/2))/xu.utilities_noconf.wavelength(emission)
	P=1+np.cos(np.radians(twotheta_deg))**2
	lowerbound=2*vects[np.argmax(ycoh)]
	yobs*=vects**2/P;ycoh*=vects**2/P
	energy=xu.utilities_noconf.energy(emission)

	for i,value in enumerate(atoms):
		if isinstance(value,str):
			atoms[i]=xu.materials.atom.Atom(value[0]+value[1:].lower(),1)

	#Berechnung der inelastischen Streuung J, Korrektur von yobs
	argsJ=np.where(vects[1:]>0.6)
	if inelcor:
		err={}
		params=lm.Parameters()
		params.add('J',1,min=0)
		def VonkTfitfunc(params):
			prmT=params.valuesdict()
			return T(vects,atoms,energy,yobs,prmT['J'])[argsJ]-T(vects,atoms,energy,yobs,prmT['J'])[argsJ][-1]
		resultT=lm.minimize(VonkTfitfunc,params,method='least_squares')
		prmT=resultT.params.valuesdict()
		for key in resultT.params:
			err[key]=resultT.params[key].stderr
		# ~ resultT.params.pretty_print()
		yobs-=prmT['J']*vects**2
		J=uq(prmT['J'],pq.dimensionless,err['J'])
	else:
		J=uq(0,pq.dimensionless,0)

	#Normierung auf elektronische Einheiten eA^-2
	normEU=np.median((fsquared(vects,atoms,energy)*vects**2)[-10:])/np.median(yobs[-10:])
	yobs*=normEU;ycoh*=normEU

	#Berechnung von Rulands R, Anpassung durch Vonks Funktion
	argsR=np.where(vects[1:]>lowerbound)
	RulandR=R(vects,yobs,ycoh)
	err={}
	params=lm.Parameters()
	params.add('C0',1,min=1)
	params.add('C1',0,min=0)
	params.add('C2',0)
	def VonkRfitfunc(params):
		prmR=params.valuesdict()
		return RulandR[argsR]-polysecond(vects**2,prmR['C0'],prmR['C1'],prmR['C2'])[argsR]
	resultR=lm.minimize(VonkRfitfunc,params,method='least_squares')
	prmR=resultR.params.valuesdict()
	for key in resultR.params:
		err[key]=resultR.params[key].stderr
	# ~ resultR.params.pretty_print()

	#Abbildungen
	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,5.3/2.54))
	ax2=ax1.twinx()

	ax1.plot(vects[argsR]**2,RulandR[argsR],'k',linewidth=0.5)
	ax1.plot(np.linspace(0,max(vects**2)),polysecond(np.linspace(0,max(vects**2)),prmR['C0'],prmR['C1'],prmR['C2']),'k--',linewidth=0.5)

	ax2.plot(vects**2,yobs,'k',linewidth=0.5)
	ax2.plot(vects**2,ycoh,'k--',linewidth=0.5)
	ax2.plot(vects**2,fsquared(vects,atoms,energy)*vects**2,'w',linewidth=0.5)
	ax2.plot(vects**2,fsquared(vects,atoms,energy)*vects**2,'k:',linewidth=0.5)

	ax1.set_xlim([0,None])
	ax1.set_ylim([0,None])
	plotlim=2*np.median((fsquared(vects,atoms,energy)*vects**2)[-10:])
	if ax2.get_ylim()[-1]>plotlim:
		ax2.set_ylim([0,None])
	else:
		ax2.set_ylim([0,plotlim])

	ax1.set_xlabel(r'$s_p^2/\rm{\AA}^{-2}$',fontsize=10)
	ax1.set_ylabel(r'$R/1$',fontsize=10)
	ax2.set_ylabel(r'$Is^2/(\rm{e\,\AA}^{-2})$',fontsize=10)
	ax1.tick_params(axis='both',pad=2,labelsize=8)
	ax2.tick_params(axis='y',pad=2,labelsize=8)
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	ax2.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_Vonk.png',dpi=300)

	xc=1/uq(prmR['C0'],pq.dimensionless,err['C0'])
	k=2*xc*uq(prmR['C1'],pq.angstrom**2,err['C1'])

	return xc,k,J
