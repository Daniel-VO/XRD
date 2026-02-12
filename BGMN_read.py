"""
Created 12. Februar 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import quantities as pq
from quantities import UncertainQuantity as uq
import BGMN_Vonk

for f in glob.glob('*.str'):
	print('Kontrolle Grenzen Gitterparameter:')
	print(f)
	l=open(f).readlines()
	for linenumber,line in enumerate(l):
		if 'PARAM=' in line and 'RP=' not in line and 'amorphous' not in f and 'single' not in f:
			for j in line.split(' '):
				print(j)
				for k in j.split('=')[2:]:
					if '_' in k and '^' in k:
						print(round((float(k.split('_')[0])/float(k.split('_')[1].split('^')[0])-1)*100,1))
						print(round((float(k.split('_')[0])/float(k.split('_')[1].split('^')[1])-1)*100,1))
					else:
						print('Grenze Gitterparameter nicht gesetzt - bitte pruefen!')

if len(sys.argv)==4:
	switch,varslitcor=sys.argv[1],sys.argv[2],eval(sys.argv[3])
else:
	switch=input('hetero oder homo [homo]? ')
	if switch=='':
		switch='homo'
	varslitcor=input('Korrektur fuer variable Blende [False]? ')
	if varslitcor=='':
		varslitcor=False
	else:
		varslitcor=eval(varslitcor)

fnlist=[]
phaselist=[]
XrayDensity_c=[]
lata_c=[]
latb_c=[]
latc_c=[]
alpha_c=[]
beta_c=[]
gamma_c=[]
GrainSize100_c=[]
GrainSize010_c=[]
GrainSize001_c=[]
MicroStrain100_c=[]
MicroStrain010_c=[]
MicroStrain001_c=[]
Textur100_c=[]
Textur010_c=[]
Textur001_c=[]
TDS100_c=[]
TDS010_c=[]
TDS001_c=[]
Gewicht_c=[]
xc_c=[]
k_c=[]
J_c=[]

for f in glob.glob('*.lst'):
	fn=os.path.splitext(f)[0]
	emission='CuKa1'
	if ('LAMBDA=cu' or 'LAMBDA=CU') in open(fn+'.sav').read():
		emission='CuKa1'
		print('Emission erkannt: '+emission)
	else:
		print('Emission nicht erkannt, falle zurück auf: '+emission)

	tt_deg,yobs,yfit,yinc=np.genfromtxt(fn+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=(0,1,2,3))
	dia=open(fn+'.dia').readlines()
	for d in range(int(dia[0].split('[')[-1].split(']')[0])):
		if 'amorph' in dia[0].split('STRUC['+str(1+d)+']=')[1].split(' STRUC[')[0].replace('\n',''):
			yinc+=np.genfromtxt(fn+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=4+d)

	l=open(f).readlines()
	for linenumber,line in enumerate(l):
		if 'Local parameters and GOALs for phase ' in line and 'amorphous' not in line and 'single' not in line:
			fnlist.append(fn)
			phasename=line.split('GOALs for phase ')[1].replace('\n','')
			phaselist.append(phasename)
		split0=line.split('=')
		if split0[0]=='UNIT':
			if 'NM' in split0[1]:
				unitoflength=pq.nm
			lata=latb=latc=uq(0,unitoflength,0)
			alpha=beta=gamma=uq(90,pq.degrees,0)
			TDS100=TDS010=TDS001=TDS=uq(0,unitoflength**2,0)
			GrainSize100=GrainSize010=GrainSize001=uq(0,unitoflength,0)
			MicroStrain100=MicroStrain010=MicroStrain001=uq(0,pq.CompoundUnit('m/m'),0)
			Textur100=Textur010=Textur001=Gewicht=uq(1,pq.dimensionless,0)
		if split0[0]=='XrayDensity':
			XrayDensity0=float(split0[1])
		if split0[0]=='A':
			if '+-' in split0[1]:
				lata=uq(float(split0[1].split('+-')[0]),unitoflength,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				lata=uq(float(split0[1]),unitoflength,0)
			latb=latc=lata
		if split0[0]=='B':
			if '+-' in split0[1]:
				latb=uq(float(split0[1].split('+-')[0]),unitoflength,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				latb=uq(float(split0[1]),unitoflength,0)
		if split0[0]=='C':
			if '+-' in split0[1]:
				latc=uq(float(split0[1].split('+-')[0]),unitoflength,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				latc=uq(float(split0[1]),unitoflength,0)
		if split0[0]=='ALPHA':
			if '+-' in split0[1]:
				alpha=uq(float(split0[1].split('+-')[0]),pq.degrees,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				alpha=uq(float(split0[1]),pq.degrees,0)
		if split0[0]=='BETA':
			if '+-' in split0[1]:
				beta=uq(float(split0[1].split('+-')[0]),pq.degrees,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				beta=uq(float(split0[1]),pq.degrees,0)
		if split0[0]=='GAMMA':
			if '+-' in split0[1]:
				gamma=uq(float(split0[1].split('+-')[0]),pq.degrees,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				gamma=uq(float(split0[1]),pq.degrees,0)
		if split0[0]=='GrainSize(1,0,0)':
			if '+-' in split0[1]:
				GrainSize100=uq(float(split0[1].split('+-')[0]),unitoflength,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				GrainSize100=uq(float(split0[1]),unitoflength,-1)
		if split0[0]=='GrainSize(0,1,0)':
			if '+-' in split0[1]:
				GrainSize010=uq(float(split0[1].split('+-')[0]),unitoflength,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				GrainSize010=uq(float(split0[1]),unitoflength,-1)
		if split0[0]=='GrainSize(0,0,1)':
			if '+-' in split0[1]:
				GrainSize001=uq(float(split0[1].split('+-')[0]),unitoflength,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				GrainSize001=uq(float(split0[1]),unitoflength,-1)
		if split0[0]=='sqrt(k2(1,0,0))':
			if '+-' in split0[1]:
				MicroStrain100=uq(float(split0[1].split('+-')[0]),pq.CompoundUnit('m/m'),float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				MicroStrain100=uq(float(split0[1]),pq.CompoundUnit('m/m'),-1)
		if split0[0]=='sqrt(k2(0,1,0))':
			if '+-' in split0[1]:
				MicroStrain010=uq(float(split0[1].split('+-')[0]),pq.CompoundUnit('m/m'),float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				MicroStrain010=uq(float(split0[1]),pq.CompoundUnit('m/m'),-1)
		if split0[0]=='sqrt(k2(0,0,1))':
			if '+-' in split0[1]:
				MicroStrain001=uq(float(split0[1].split('+-')[0]),pq.CompoundUnit('m/m'),float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				MicroStrain001=uq(float(split0[1]),pq.CompoundUnit('m/m'),-1)
		if split0[0]=='GEWICHT(1,0,0)/GEWICHT':
			if '+-' in split0[1]:
				Textur100=uq(float(split0[1].split('+-')[0]),pq.dimensionless,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				Textur100=uq(float(split0[1]),pq.dimensionless,-1)
		if split0[0]=='GEWICHT(0,1,0)/GEWICHT':
			if '+-' in split0[1]:
				Textur010=uq(float(split0[1].split('+-')[0]),pq.dimensionless,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				Textur010=uq(float(split0[1]),pq.dimensionless,-1)
		if split0[0]=='GEWICHT(0,0,1)/GEWICHT':
			if '+-' in split0[1]:
				Textur001=uq(float(split0[1].split('+-')[0]),pq.dimensionless,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				Textur001=uq(float(split0[1]),pq.dimensionless,-1)
		if split0[0]=='TDS(1,0,0)':
			if '+-' in split0[1]:
				TDS100=uq(float(split0[1].split('+-')[0]),unitoflength**2,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				TDS100=uq(float(split0[1]),unitoflength**2,-1)
		if split0[0]=='TDS(0,1,0)':
			if '+-' in split0[1]:
				TDS010=uq(float(split0[1].split('+-')[0]),unitoflength**2,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				TDS010=uq(float(split0[1]),unitoflength**2,-1)
		if split0[0]=='TDS(0,0,1)':
			if '+-' in split0[1]:
				TDS001=uq(float(split0[1].split('+-')[0]),unitoflength**2,float(split0[1].split('+-')[1]))
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				TDS001=uq(float(split0[1]),unitoflength**2,-1)
		if split0[0]=='GEWICHT':
			if '+-' in split0[1]:
				Gewicht=uq(float(split0[1].split('+-')[0]),pq.dimensionless,float(split0[1].split('+-')[1]))
			elif 'MeanValue(GEWICHT)' in split0[1]:
				Gewicht=uq(float(split0[2]),pq.dimensionless,0)
			elif 'UNDEF' not in split0[1] and 'ERROR' not in split0[1]:
				Gewicht=uq(float(split0[1]),pq.dimensionless,0)
			else:
				Gewicht=uq(0,pq.dimensionless,0)

		if 'Atomic positions for phase' in line and 'amorphous' not in line and 'single' not in line:
			atoms_c,occups_c=[],[]
			for linenumber1,line1 in enumerate(l[linenumber:]):
				if 'E=' in line1:
					atoms_c.append(line1.split('=(')[1].split('(')[0].split('+')[0].split('-')[0])
					occups_c.append(float(line1.split('=(')[1].split('(')[1].split(')')[0]))
				if 'Local parameters and GOALs for phase' in line1:
					break
			atoms=[]
			atoms_c,occups_c=np.array(atoms_c),np.array(occups_c)
			for a in np.unique(atoms_c):
				numbers=int(round(np.sum(occups_c[atoms_c==a]),0))
				for j in range(numbers):
					atoms.append(str(a))
			for d in range(int(dia[0].split('[')[-1].split(']')[0])):
				if dia[0].split('STRUC['+str(1+d)+']=')[1].split(' STRUC[')[0].replace('\n','')==phasename:
					ycoh=np.genfromtxt(fn+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=4+d)
				if 'single' in dia[0].split('STRUC['+str(1+d)+']=')[1].split(' STRUC[')[0].replace('\n',''):
					ycoh+=np.genfromtxt(fn+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=4+d)
			if np.median(ycoh)!=0:
				if switch=='homo':
					xc,k,J=BGMN_Vonk.Vonk(fn+'_'+phasename,atoms,yobs*1,ycoh,tt_deg,emission,varslitcor)
				elif switch=='hetero':
					xc,k,J=BGMN_Vonk.Vonk(fn+'_'+phasename,atoms,yinc+ycoh,ycoh,tt_deg,emission,varslitcor)
					print('Warnung: xc ist kristalliner Anteil an homogener Portion.')
				else:
					print('Eingabe hetero / homo nicht verstanden, xc wird auf 0 gesetzt.')
					xc,k,J=uq(0,pq.dimensionless,0),uq(0,pq.angstrom**2,0),uq(0,pq.dimensionless,0)
			else:
				xc,k,J=uq(0,pq.dimensionless,0),uq(0,pq.angstrom**2,0),uq(0,pq.dimensionless,0)

			Vol=lata*latb*latc*(1-np.cos(np.radians(alpha))**2-np.cos(np.radians(beta))**2-np.cos(np.radians(gamma))**2+2*np.cos(np.radians(alpha))*np.cos(np.radians(beta))*np.cos(np.radians(gamma)))**0.5
			XrayDensity_c.append(uq(XrayDensity0,pq.kg/pq.l,float(Vol.uncertainty/Vol.magnitude)))
			lata_c.append(lata)
			latb_c.append(latb)
			latc_c.append(latc)
			alpha_c.append(alpha)
			beta_c.append(beta)
			gamma_c.append(gamma)
			GrainSize100_c.append(GrainSize100)
			GrainSize010_c.append(GrainSize010)
			GrainSize001_c.append(GrainSize001)
			MicroStrain100_c.append(MicroStrain100)
			MicroStrain010_c.append(MicroStrain010)
			MicroStrain001_c.append(MicroStrain001)
			Textsum=np.sum([Textur100,Textur010,Textur001])
			Textur100_c.append(Textur100/Textsum)
			Textur010_c.append(Textur010/Textsum)
			Textur001_c.append(Textur001/Textsum)
			TDS100_c.append(TDS100)
			TDS010_c.append(TDS010)
			TDS001_c.append(TDS001)
			Gewicht_c.append(Gewicht)
			xc_c.append(xc)
			k_c.append(k)
			J_c.append(J)

export=[fnlist,phaselist,XrayDensity_c,lata_c,latb_c,latc_c,alpha_c,beta_c,gamma_c,GrainSize100_c,GrainSize010_c,GrainSize001_c,MicroStrain100_c,MicroStrain010_c,MicroStrain001_c,Textur100_c,Textur010_c,Textur001_c,TDS100_c,TDS010_c,TDS001_c,Gewicht_c,xc_c,k_c,J_c]

# ~ print(export)

def namestr(obj, namespace):
	return str([name for name in namespace if namespace[name] is obj][0])
exportstring=str()
for j,value in enumerate(export):
	if j!=0:
		exportstring+=','
	exportstring+=namestr(export[j],locals())

os.system('mv '+'results.pic '+'results_alt.pic')
os.system('mv '+'results.txt '+'results_alt.txt')

pickle.dump(export,open('results.pic','wb'))

print('____')
print('Ausgegeben als Liste von Python quantities.UncertainQuantity: ['+exportstring+']')
print('____')
print("Zum Laden der Liste: pickle.load(open('results.pic','rb')")

for f,valuei in enumerate(fnlist):
	printline=str(fnlist[f])+'; '+str(phaselist[f])
	for j,valuej in enumerate(export):
		if j>1 and valuej!=[]:
			printline+='; '+namestr(export[j],locals())+': '+'%.8e'%float(valuej[f].magnitude)+' +/- '+'%.8e'%valuej[f].uncertainty
	printline+='; comp: '+switch+' varslitcor: '+str(varslitcor)
	print(printline,file=open('results.txt','a'))
