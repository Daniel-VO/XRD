"""
Created 12. January 2021 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy
import glob
import os
import fnmatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

os.system('mv Read.log Read.alt')

data=numpy.load('data.npy')
#[filename,R,A,E,EOF,Ag,EOFpl]

nameslist=[]
for n,value in enumerate(data):
	print(value[0])
	nameslist.append(value[0].split('/')[1][:-3])
	print(value[0].split('/')[1][:-3])

def common_elements(list1, list2):
	return list(set(list1) & set(list2))

samples=common_elements(nameslist,nameslist)

sampledata=[]

for m,name in enumerate(samples):
	R=numpy.array([])
	A=numpy.array([])
	E=numpy.array([])
	EOF=numpy.array([])
	Ag=numpy.array([])
	EOFpl=numpy.array([])
	for n,value in enumerate(data):
		if value[0].split('/')[1][:-3]==name:
			R=numpy.append(R,float(value[1]))
			A=numpy.append(A,float(value[2]))
			E=numpy.append(E,float(value[3]))
			EOF=numpy.append(EOF,float(value[4]))
			Ag=numpy.append(Ag,float(value[5]))
			EOFpl=numpy.append(EOFpl,float(value[6]))
	sampledata.append([name.replace('_','-'),'R:',numpy.average(R),'pm',numpy.std(R),'A:',numpy.average(A),'pm',numpy.std(A),'E:',numpy.average(E),'pm',numpy.std(E),'EOF:',numpy.average(EOF),'pm',numpy.std(EOF),'Ag:',numpy.average(Ag),'pm',numpy.std(Ag),'EOFpl:',numpy.average(EOFpl),'pm',numpy.std(EOFpl)])

for i,value in enumerate(sampledata):
	with open('Read.log','a') as e:
		print(e,sampledata[i])

sets=['']

for s in sets:
	nameplot=[]
	Rplot=[]
	Rstdplot=[]
	Aplot=[]
	Astdplot=[]
	Eplot=[]
	Estdplot=[]
	Econfloplot=[]
	Econfupplot=[]
	EOFplot=[]
	EOFstdplot=[]
	Agplot=[]
	Agstdplot=[]
	EOFplplot=[]
	EOFplstdplot=[]
	for m,name in enumerate(sampledata):
		if s in sampledata[m][0]:
			nameplot.append(sampledata[m][0])
			Rplot.append(sampledata[m][2])
			Rstdplot.append(sampledata[m][4])
			Aplot.append(sampledata[m][6])
			Astdplot.append(sampledata[m][8])
			Eplot.append(sampledata[m][10])
			Estdplot.append(sampledata[m][12])
			EOFplot.append(sampledata[m][14])
			EOFstdplot.append(sampledata[m][16])
			Agplot.append(sampledata[m][18])
			Agstdplot.append(sampledata[m][20])
			EOFplplot.append(sampledata[m][22])
			EOFplstdplot.append(sampledata[m][24])
			xlabel=r'$\rm{Probe}$'

	########################################################################

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,7.5/2.54))

	ax1.errorbar(nameplot,Rplot,marker='s',color='k',yerr=Rstdplot,markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0)

	ax1.set_xlabel(xlabel,fontsize=10)
	plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,ha='right',rotation_mode='anchor')
	ax1.set_ylabel(r'$R/\rm{Pa}$',fontsize=10)
	ax1.tick_params(direction='out')
	ax1.tick_params(axis='x',pad=2,labelsize=8)
	ax1.tick_params(axis='y',pad=2,labelsize=8)
	ax1.ticklabel_format(style='sci',axis='y',scilimits=(-3,3))
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig('R'+s+'.pdf',transparent=True)
	plt.savefig('R'+s+'.png',dpi=600)
	plt.close('all')

	########################################################################

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,7.5/2.54))

	ax1.errorbar(nameplot,Aplot,marker='s',color='k',yerr=Astdplot,markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0)
	ax1.errorbar(nameplot,Agplot,marker='o',color='k',yerr=Agstdplot,markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0)

	ax1.set_xlabel(xlabel,fontsize=10)
	plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,ha='right',rotation_mode='anchor')
	ax1.set_ylabel(r'$A/1$',fontsize=10)
	ax1.tick_params(direction='out')
	ax1.tick_params(axis='x',pad=2,labelsize=8)
	ax1.tick_params(axis='y',pad=2,labelsize=8)
	ax1.ticklabel_format(style='sci',axis='y',scilimits=(-3,3))
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig('A'+s+'.pdf',transparent=True)
	plt.savefig('A'+s+'.png',dpi=600)
	plt.close('all')

	########################################################################

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,7.5/2.54))

	ax1.errorbar(nameplot,Eplot,marker='s',color='k',yerr=Estdplot,markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0)

	ax1.set_xlabel(xlabel,fontsize=10)
	plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,ha='right',rotation_mode='anchor')
	ax1.set_ylabel(r'$E/\rm{Pa}$',fontsize=10)
	ax1.tick_params(direction='out')
	ax1.tick_params(axis='x',pad=2,labelsize=8)
	ax1.tick_params(axis='y',pad=2,labelsize=8)
	ax1.ticklabel_format(style='sci',axis='y',scilimits=(-3,3))
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig('E'+s+'.pdf',transparent=True)
	plt.savefig('E'+s+'.png',dpi=600)
	plt.close('all')

	########################################################################

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,7.5/2.54))

	ax1.errorbar(nameplot,EOFplot,marker='s',color='k',yerr=EOFstdplot,markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0)
	ax1.errorbar(nameplot,EOFplplot,marker='o',color='k',yerr=EOFplstdplot,markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0)

	ax1.set_xlabel(xlabel,fontsize=10)
	plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,ha='right',rotation_mode='anchor')
	ax1.set_ylabel(r'$U_{\rm{f}}/\rm{Jm}^{-3}$',fontsize=10)
	ax1.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	ax1.tick_params(direction='out')
	ax1.tick_params(axis='x',pad=2,labelsize=8)
	ax1.tick_params(axis='y',pad=2,labelsize=8)
	ax1.ticklabel_format(style='sci',axis='y',scilimits=(-3,3))
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig('EOF'+s+'.pdf',transparent=True)
	plt.savefig('EOF'+s+'.png',dpi=600)
	plt.close('all')
