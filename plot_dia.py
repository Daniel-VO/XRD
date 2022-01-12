import glob
import numpy
import string
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

for i in glob.glob('*.dia'):
	filename=os.path.splitext(i)[0]
	data=numpy.genfromtxt(i,unpack=True,skip_header=1)
	columns=['twotheta','yobs','yfit','ybkg']
	for j in open(i).readline().split(' '):
		if 'STRUC' in j:
			columns.append(j.split('=')[1].split('\n')[0])

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5/2.54,5.3/2.54))

	linestyles=['k--','k-.','k:']
	for k,name in enumerate(columns):
		if 'amorph' in name:
			ybkg+=data[k]
		else:
			vars()[name]=data[k]
			if k>3:
				ax1.plot(twotheta,data[k],linestyles[k-4],linewidth=0.5)

	ax1.plot(twotheta,yobs,'k',linewidth=1)
	ax1.plot(twotheta,yfit,'w',linewidth=0.5)
	ax1.plot(twotheta,yfit,'k',linewidth=0.15)
	ax1.plot(twotheta,ybkg,'0.5',linewidth=0.5)

	if '' in filename:
		plt.figtext(0,0.95,r'$\rm{(a)}$',fontsize=10)

	ax1.set_yticks([0])

	ax1.set_xlabel(r'$2\theta/^\circ$',fontsize=10)
	ax1.set_ylabel(r'$I/1$',fontsize=10)
	ax1.tick_params(direction='out')
	ax1.tick_params(axis='x',pad=2,labelsize=8)
	ax1.tick_params(axis='y',pad=2,labelsize=8)
	ax1.xaxis.get_offset_text().set_size(8)
	ax1.yaxis.get_offset_text().set_size(8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'.pdf',transparency=True)
	plt.savefig(filename+'.png',dpi=600)
	plt.close('all')
