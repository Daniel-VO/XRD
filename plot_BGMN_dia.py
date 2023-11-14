"""
Created 14.November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import glob
import numpy as np
import string
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

for i in glob.glob('*.dia'):
	filename=os.path.splitext(i)[0]
	data=np.genfromtxt(i,unpack=True,skip_header=1)
	columns=['twotheta','yobs','yfit','ybkg']
	for j in open(i).readline().split(' '):
		if 'STRUC' in j:
			columns.append(j.split('=')[1].split('\n')[0])

	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	plt.subplots(figsize=(7.5/2.54,5.3/2.54))

	linestyles=['k--','k-.','k:']
	for k,name in enumerate(columns):
		if 'amorph' in name:
			ybkg+=data[k]
		else:
			vars()[name]=data[k]
			if k>3:
				plt.plot(twotheta,data[k],linestyles[k-4],linewidth=0.5)

	plt.plot(twotheta,yobs,'k',linewidth=1)
	plt.plot(twotheta,yfit,'w',linewidth=0.5)
	plt.plot(twotheta,yfit,'k',linewidth=0.15)
	plt.plot(twotheta,ybkg,'0.5',linewidth=0.5)

	if '' in filename:
		plt.figtext(0,0.95,r'$\rm{(a)}$',fontsize=10)

	plt.yticks([0])

	plt.xlabel(r'$2\theta/^\circ$',fontsize=10)
	plt.ylabel(r'$I/1$',fontsize=10)
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'.pdf',transparent=True)
	plt.savefig(filename+'.png',dpi=300)
