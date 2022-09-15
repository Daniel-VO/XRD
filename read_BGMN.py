"""
Created 15. September 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy
import os
import sys
import glob
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import quantities as pq
from quantities import UncertainQuantity as uq
from scipy import integrate
import Vonk

for i in glob.glob('*.str'):
	print('Kontrolle Grenzen Gitterparameter:')
	print(i)
	f=open(i).readlines()
	for linenumber,line in enumerate(f):
		if 'PARAM=' in line and 'RP=' not in line and 'amorphous' not in i:
			for j in line.split(' '):
				print(j)
				for k in j.split('=')[2:]:
					if '_' in k and '^' in k:
						print(round((float(k.split('_')[0])/float(k.split('_')[1].split('^')[0])-1)*100,1))
						print(round((float(k.split('_')[0])/float(k.split('_')[1].split('^')[1])-1)*100,1))
					else:
						print('Grenze Gitterparameter nicht gesetzt - bitte pruefen!')

if len(sys.argv)==5:
	filenamepattern,switch,lowerbound,incohcor=sys.argv[1],sys.argv[2],float(sys.argv[3]),eval(sys.argv[4])
else:
	filenamepattern=input('Muster Dateinamen [*]: ')
	if filenamepattern=='':
		filenamepattern='*'
	switch=input('hetero oder homo [homo]? ')
	if switch=='':
		switch='homo'
	lowerbound=input('Untere Integrationsgrenze in A^-1 [0.3]: ')
	if lowerbound=='':
		lowerbound=0.3
	else:
		lowerbound=float(lowerbound)
	incohcor=input('Korrektur fuer inkohaerente Streuung [False]? ')
	if incohcor=='':
		incohcor=False
	else:
		incohcor=eval(incohcor)

filenamelist=[]
phaselist=[]
XrayDensity_collect=[]
lata_collect=[]
latb_collect=[]
latc_collect=[]
GrainSize100_collect=[]
GrainSize010_collect=[]
GrainSize001_collect=[]
MicroStrain100_collect=[]
MicroStrain010_collect=[]
MicroStrain001_collect=[]
Textur100_collect=[]
Textur010_collect=[]
Textur001_collect=[]
TDS100_collect=[]
TDS010_collect=[]
TDS001_collect=[]
Gewicht_collect=[]
fc_collect=[]
k_collect=[]
J_collect=[]

for i in glob.glob(filenamepattern+'.lst'):
	filename=os.path.splitext(i)[0]
	print(filename)

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,5.3/2.54))

	emission='CuKa1'
	if ('LAMBDA=cu' or 'LAMBDA=CU') in open(filename+'.sav').read():
		emission='CuKa1'
		print('Emission erkannt: '+emission)
	else:
		print('Emission nicht erkannt, falle zurück auf: '+emission)

	twotheta,yobs,yfit,yam=numpy.genfromtxt(filename+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=(0,1,2,3))
	dia=open(filename+'.dia').readlines()
	for d in numpy.arange(int(dia[0].split('[')[-1].split(']')[0])):
		if 'amorph' in dia[0].split('STRUC['+str(1+d)+']=')[1].split(' STRUC[')[0].replace('\n',''):
			yam+=numpy.genfromtxt(filename+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=4+d)

	f=open(i).readlines()
	for linenumber,line in enumerate(f):
		if 'Local parameters and GOALs for phase ' in line and 'amorphous' not in line:
			filenamelist.append(filename)
			phasename=line.split('GOALs for phase ')[1].replace('\n','')
			phaselist.append(phasename)
		split0=line.split('=')
		if split0[0]=='UNIT':
			if 'NM' in split0[1]:
				unitoflength=pq.nm
			lata=latb=latc=uq(0,unitoflength,0)
			TDS100=TDS010=TDS001=TDS=uq(0,unitoflength**2,0)
			GrainSize100=GrainSize010=GrainSize001=uq(0,unitoflength,0)
			MicroStrain100=MicroStrain010=MicroStrain001=uq(0,pq.CompoundUnit('m/m'),0)
			Textur100=Textur010=Textur001=Gewicht=uq(1,pq.dimensionless,0)
		####
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

		if 'Atomic positions for phase' in line and 'amorphous' not in line:
			Vol=lata*latb*latc
			XrayDensity=uq(XrayDensity0,pq.kg/pq.l,float(Vol.uncertainty/Vol.magnitude))
			####
			atoms_collect,occups_collect=[],[]
			for linenumber1,line1 in enumerate(f[linenumber:]):
				if 'E=' in line1:
					atoms_collect.append(line1.split('=(')[1].split('(')[0].split('+')[0].split('-')[0])
					occups_collect.append(float(line1.split('=(')[1].split('(')[1].split(')')[0]))
				if 'Local parameters and GOALs for phase' in line1:
					break
			atoms=[]
			atoms_collect,occups_collect=numpy.array(atoms_collect),numpy.array(occups_collect)
			for a in numpy.unique(atoms_collect):
				numbers=int(round(numpy.sum(occups_collect[numpy.where(atoms_collect==a)]),0))
				for j in numpy.arange(numbers):
					atoms.append(str(a))
			for d in numpy.arange(int(dia[0].split('[')[-1].split(']')[0])):
				if dia[0].split('STRUC['+str(1+d)+']=')[1].split(' STRUC[')[0].replace('\n','')==phasename:
					col=4+d
			ycryst=numpy.genfromtxt(filename+'.dia',delimiter=None,unpack=True,skip_header=1,skip_footer=0,usecols=col)
			if numpy.median(ycryst)!=0:
				if switch=='homo':
					fc,k,J=Vonk.Vonk(filename+'_'+phasename,atoms,yobs*1,ycryst,twotheta,emission,True,lowerbound,incohcor)
				elif switch=='hetero':
					fc,k,J=Vonk.Vonk(filename+'_'+phasename,atoms,yam+ycryst,ycryst,twotheta,emission,True,lowerbound,incohcor)
					print('Warnung: fc ist kristalliner Anteil an homogener Portion.')
				else:
					print('Eingabe hetero / homo nicht verstanden, fc wird auf 0 gesetzt.')
					fc,k,J=uq(0,pq.dimensionless,0),uq(0,pq.angstrom**2,0),uq(0,pq.dimensionless,0)
			else:
				fc,k,J=uq(0,pq.dimensionless,0),uq(0,pq.angstrom**2,0),uq(0,pq.dimensionless,0)

			####
			XrayDensity_collect.append(XrayDensity)
			lata_collect.append(lata)
			latb_collect.append(latb)
			latc_collect.append(latc)
			GrainSize100_collect.append(GrainSize100)
			GrainSize010_collect.append(GrainSize010)
			GrainSize001_collect.append(GrainSize001)
			MicroStrain100_collect.append(MicroStrain100)
			MicroStrain010_collect.append(MicroStrain010)
			MicroStrain001_collect.append(MicroStrain001)
			Textur100_collect.append(Textur100)
			Textur010_collect.append(Textur010)
			Textur001_collect.append(Textur001)
			TDS100_collect.append(TDS100)
			TDS010_collect.append(TDS010)
			TDS001_collect.append(TDS001)
			Gewicht_collect.append(Gewicht)
			fc_collect.append(fc)
			k_collect.append(k)
			J_collect.append(J)

export=[filenamelist,phaselist,XrayDensity_collect,lata_collect,latb_collect,latc_collect,GrainSize100_collect,GrainSize010_collect,GrainSize001_collect,MicroStrain100_collect,MicroStrain010_collect,MicroStrain001_collect,Textur100_collect,Textur010_collect,Textur001_collect,TDS100_collect,TDS010_collect,TDS001_collect,Gewicht_collect,fc_collect,k_collect,J_collect]

# ~ print(export)

def namestr(obj, namespace):
	return str([name for name in namespace if namespace[name] is obj][0])
exportstring=str()
for j,value in enumerate(export):
	if j!=0:
		exportstring+=','
	exportstring+=namestr(export[j],locals())

os.system('mv '+filenamepattern.replace('*','alle')+'.pic '+filenamepattern.replace('*','alle')+'_alt.pic')
os.system('mv '+filenamepattern.replace('*','alle')+'.txt '+filenamepattern.replace('*','alle')+'_alt.txt')

pickle.dump(export,open(filenamepattern.replace('*','alle')+'.pic','wb'))

print('____')
print('Ausgegeben als Liste von Python quantities.UncertainQuantity: ['+exportstring+']')
print('____')
print("Zum Laden der Liste: pickle.load(open(filenamepattern+'.pic','rb')")

sys.stdout=open(filenamepattern.replace('*','alle')+'.txt','w')
for i,valuei in enumerate(filenamelist):
	printline=str(filenamelist[i])+'; '+str(phaselist[i])
	for j,valuej in enumerate(export):
		if j>1 and valuej!=[]:
			printline+='; '+namestr(export[j],locals())+': '+str(float(valuej[i].magnitude))+' +/- '+str(valuej[i].uncertainty)
	print(printline)
