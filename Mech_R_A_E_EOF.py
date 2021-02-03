"""
Created 14. January 2021 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy
import glob
import os
import sys
import fnmatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import matplotlib.patheffects as pe

def conv(t):
	return t.replace(',','.')

os.system('mv Results.log Results.alt')

files=[]
data=[]

for root, dirnames, filenames in os.walk('.'):
	for filename in fnmatch.filter(filenames, '*.txt'):
		files.append(os.path.join(root,filename))

for i in files:
	filename=os.path.splitext(i)[0]

	sys.stdout=open('Results.log','a')

	Zeit_s,Kraft_N,Weg_mm,Spannung_MPa,Dehnung_perc=numpy.genfromtxt((conv(t) for t in open(i)),delimiter='\t',unpack=True,skip_header=1,skip_footer=0,usecols=range(5))
	Spannung=(Spannung_MPa-Spannung_MPa[0])*1e6
	Dehnung=(Dehnung_perc-Dehnung_perc[0])/1e2

	Punkte=1000		#Punkte pro Segment zur Bestimmung von E

	args=numpy.where(Spannung>numpy.pi*numpy.median(abs(numpy.diff(Spannung))))[0]
	if len(args)>2*Punkte:
		Spannung=Spannung[args]
		Dehnung=Dehnung[args]
	Spannung-=min(Spannung)
	Dehnung-=min(Dehnung)

	R=max(Spannung)

	Agleichmass=float(Dehnung[numpy.where(Spannung==R)][0])
	Abruch=float(Dehnung[numpy.where(Spannung>=R/10)][::-1][0])

	steps=int(len(Dehnung[numpy.where(Dehnung<Agleichmass)])/Punkte)
	theils=numpy.array([])
	for i in numpy.arange(steps):
		ind=numpy.where((Dehnung>=i*Agleichmass/steps)&(Dehnung<(1+i)*Agleichmass/steps))
		theils=numpy.append(theils,stats.theilslopes(Spannung[ind],Dehnung[ind],0.9))
	theils=theils.reshape(steps,4)
	theil=theils[numpy.where(theils[:,0]==max(theils[:,0]))][0]
	E=theil[0]
	disp=theil[1]
	Econflo=theil[2]
	Econfup=theil[3]

	A=Abruch-R/E
	Ag=Agleichmass-R/E

	EOF=numpy.trapz(Spannung[numpy.where(Dehnung<=Abruch)],x=Dehnung[numpy.where(Dehnung<=Abruch)])
	EOFpl=EOF-R**2/E/2

	data.append([filename,R,A,E,EOF,Ag,EOFpl])

	print(filename,'R',R,'A',A,'E',E,'EOF',EOF,'Ag',Ag,'EOFpl',EOFpl)

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,5.3/2.54))

	ax1.plot(Dehnung[numpy.where((Dehnung>=A)&(Dehnung<=Abruch))],(Dehnung[numpy.where((Dehnung>=A)&(Dehnung<=Abruch))]+A)*E-numpy.min((Dehnung[numpy.where((Dehnung>=A)&(Dehnung<=Abruch))]+A)*E),'k--',linewidth=0.5,path_effects=[pe.Stroke(linewidth=1,foreground='white'),pe.Normal()])
	plt.plot(Dehnung[numpy.where(Dehnung<=Agleichmass)],Dehnung[numpy.where(Dehnung<=Agleichmass)]*E+disp,'k',linewidth=0.5,path_effects=[pe.Stroke(linewidth=1,foreground='white'),pe.Normal()])
	plt.plot(Dehnung[numpy.where(Dehnung<=Agleichmass)],Dehnung[numpy.where(Dehnung<=Agleichmass)]*Econflo+disp,'k:',linewidth=0.5,path_effects=[pe.Stroke(linewidth=1,foreground='white'),pe.Normal()])
	plt.plot(Dehnung[numpy.where(Dehnung<=Agleichmass)],Dehnung[numpy.where(Dehnung<=Agleichmass)]*Econfup+disp,'k:',linewidth=0.5,path_effects=[pe.Stroke(linewidth=1,foreground='white'),pe.Normal()])
	ax1.errorbar(Abruch,R*0.25,marker='s',color='k',markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0,path_effects=[pe.Stroke(linewidth=2,foreground='w'),pe.Normal()],zorder=10)
	ax1.errorbar(Agleichmass,R,marker='s',color='k',markersize=1,elinewidth=0.5,capthick=0.5,capsize=2,linewidth=0,path_effects=[pe.Stroke(linewidth=2,foreground='w'),pe.Normal()],zorder=10)
	ax1.plot(Dehnung[numpy.where(Dehnung<=Abruch*1.5)],Spannung[numpy.where(Dehnung<=Abruch*1.5)],'k',linewidth=0.5,path_effects=[pe.Stroke(linewidth=1,foreground='white'),pe.Normal()])

	ax1.set_ylim([-R*0.05,R*1.1])

	ax1.set_xlabel(r'$\epsilon/1$',fontsize=10)
	ax1.set_ylabel(r'$\sigma/\rm{Pa}$',fontsize=10)
	ax1.tick_params(direction='out')
	ax1.tick_params(axis='x',pad=2,labelsize=8)
	ax1.tick_params(axis='y',pad=2,labelsize=8)
	ax1.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'.pdf',transparent=True)
	plt.savefig(filename+'.png',dpi=600)
	plt.close('all')

numpy.save('data.npy',data,allow_pickle=True)
os.system('python Read_Mech_R_A_E_EOF.py')
