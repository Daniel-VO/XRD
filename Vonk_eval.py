"""
Created 26. October 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import sys
import glob
import numpy
import lmfit
import scipy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import xrayutilities as xu
import quantities as pq
from quantities import UncertainQuantity as uq
from matplotlib import cm

def fsquared(vects,atoms,energy):
	return numpy.real(numpy.average(numpy.array([i.f(2*numpy.pi*vects,en=energy) for i in atoms])**2,axis=0))

def R(vects,yobs,ycoh):
	return scipy.integrate.cumtrapz(yobs,x=vects)/scipy.integrate.cumtrapz(ycoh,x=vects)

def polysecond(x,C0,C1,C2):
	return C0+C1*x+C2*x**2

@ray.remote
def calc(ciffile,rpo,sp0,L,xc0,k0):
	phasename=os.path.splitext(ciffile)[0]
	lattice=xu.materials.cif.CIFFile(ciffile).SGLattice()
	crystal=xu.materials.material.Crystal(phasename,lattice)

	atoms=['C']*str(lattice).count('C')+\
		  ['O']*str(lattice).count('O')+\
		  ['H']*str(lattice).count('H')
	for i,value in enumerate(atoms):
		if isinstance(value,str):
			atoms[i]=xu.materials.atom.Atom(value[0]+value[1:].lower(),1)

	twotheta_deg=numpy.linspace(0,180,18001)
	vects0=2*numpy.sin(numpy.radians(twotheta_deg/2))/xu.utilities_noconf.wavelength(emission)

	incmodel=xu.simpack.PowderModel(xu.simpack.Powder(crystal,1,crystallite_size_lor=1e-8,crystallite_size_gauss=2e-9,strain_gauss=20))
	yinc0=scipy.ndimage.gaussian_filter1d(incmodel.simulate(twotheta_deg),len(twotheta_deg)//36)
	incmodel.close()

	powder100=xu.simpack.Powder(crystal,1,crystallite_size_lor=5*L,crystallite_size_gauss=L,preferred_orientation=(1,0,0),preferred_orientation_factor=rpo)
	powder010=xu.simpack.Powder(crystal,1,crystallite_size_lor=5*L,crystallite_size_gauss=L,preferred_orientation=(0,1,0),preferred_orientation_factor=rpo)
	powder001=xu.simpack.Powder(crystal,1,crystallite_size_lor=5*L,crystallite_size_gauss=L,preferred_orientation=(0,0,1),preferred_orientation_factor=rpo)

	if rpo!=1:
		powderlist=[powder100,powder010,powder001]
	else:
		powderlist=[powder100]

	xc_collect,k_collect,text_collect=[],[],[]

	plt.clf()
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,axs=plt.subplots(2,figsize=(7.5/2.54,7.5/2.54))
	for i,p in enumerate(powderlist):
		cohmodel=xu.simpack.PowderModel(p)
		ycoh0=cohmodel.simulate(twotheta_deg)
		cohmodel.close()

		argsnan=numpy.where((numpy.isnan(yinc0)==False)&(numpy.isnan(ycoh0)==False))
		vects=vects0[argsnan]
		DWF=numpy.exp(-k0*vects**2)
		yinc=yinc0[argsnan]*vects**2
		ycoh=ycoh0[argsnan]*vects**2

		if i==0:
			normEU=numpy.trapz(fsquared(vects,atoms,energy)*vects**2,x=vects)/numpy.trapz(yinc,x=vects)
			axs[0].fill_between(vects,yinc*normEU*(1-xc0),yinc*normEU*(1-xc0*DWF),color='tab:gray',linewidth=0.5)
			axs[0].plot(vects,fsquared(vects,atoms,energy)*vects**2,'k--',linewidth=0.5)
		normINT=numpy.trapz(yinc,x=vects)/numpy.trapz(ycoh,x=vects)
		yinc*=normEU*(1-xc0*DWF)
		ycoh*=normEU*xc0*DWF*normINT

		argsR=numpy.where(vects[1:]>sp0)
		RulandR=R(vects,yinc+ycoh,ycoh)
		err={}
		params=lmfit.Parameters()
		params.add('C0',1,min=1)
		params.add('C1',0,min=0)
		params.add('C2',0)
		def VonkRfitfunc(params):
			prmR=params.valuesdict()
			return RulandR[argsR]-polysecond(vects**2,prmR['C0'],prmR['C1'],prmR['C2'])[argsR]
		resultR=lmfit.minimize(VonkRfitfunc,params,method='least_squares')
		prmR=resultR.params.valuesdict()
		for key in resultR.params:
			err[key]=resultR.params[key].stderr

		xc=1/uq(prmR['C0'],pq.dimensionless,err['C0'])
		k=2*xc*uq(prmR['C1'],pq.angstrom**2,err['C1'])

		if rpo!=1:
			xc_collect.append(xc),k_collect.append(k)
		else:
			xc_collect.extend((xc,xc,xc)),k_collect.extend((k,k,k))

		axs[0].plot(vects,ycoh+yinc,color=colors[i],linewidth=0.5)
		axs[1].plot(vects[1:][argsR]**2,RulandR[argsR],color=colors[i],linewidth=0.5)
		axs[1].plot(vects**2,polysecond(vects**2,prmR['C0'],prmR['C1'],prmR['C2']),color=colors[i],linewidth=0.5,linestyle='dashed')

		text_collect.append(r'$x_{\rm{c}}^{'+superscripts[i]+'}='+'%.2f'%xc.magnitude+'\quad k^{'+superscripts[i]+'}='+'%.2f'%k.magnitude+'$')

	for j,t in enumerate(text_collect):
		axs[1].text(0,axs[1].get_ylim()[1]-(axs[1].get_ylim()[1]-axs[1].get_ylim()[0])*(j+1)/10,t,fontsize=6,color=colors[j])

	plt.figtext(0,0.962,r'$\rm{(a)}$',fontsize=10)
	plt.figtext(0,0.467,r'$\rm{(b)}$',fontsize=10)

	axs[0].set_xlabel(r'$s/\rm{\AA}^{-1}$',labelpad=1,fontsize=8)
	axs[0].set_ylabel(r'$Is^2/(\rm{e\,\AA}^{-2})$',labelpad=2,fontsize=8)
	axs[1].set_xlabel(r'$s_p^2/\rm{\AA}^{-2}$',labelpad=1,fontsize=8)
	axs[1].set_ylabel(r'$R/1$',labelpad=2,fontsize=8)
	axs[0].tick_params(axis='x',pad=1,labelsize=7)
	axs[0].tick_params(axis='y',pad=1,labelsize=7)
	axs[1].tick_params(axis='x',pad=1,labelsize=7)
	axs[1].tick_params(axis='y',pad=1,labelsize=7)
	plt.tight_layout(pad=0.1)

	plt.savefig(phasename+'_'+('%.2f'%rpo).replace('.','-')+'_'+('%.1f'%sp0).replace('.','-')+'_'+'%.2i'%(L*1e9)+'_'+('%.2f'%xc0).replace('.','-')+'_'+'%.i'%k0+'.png',dpi=300)
	plt.savefig(phasename+'_'+('%.2f'%rpo).replace('.','-')+'_'+('%.1f'%sp0).replace('.','-')+'_'+'%.2i'%(L*1e9)+'_'+('%.2f'%xc0).replace('.','-')+'_'+'%.i'%k0+'.pdf',transparent=True)
	plt.close('all')

	return [ciffile,rpo,sp0,L,xc0,k0,xc_collect,k_collect]

emission='CuKa1'
energy=xu.utilities_noconf.energy(emission)
ciffile=glob.glob('*.cif')
rpo=[0.5,0.8,1.0,1.25,2]
sp0=numpy.linspace(0.3,0.9,3)
L=numpy.logspace(0,1,3)*5e-9
xc0=numpy.linspace(0.25,0.75,3)
k0=numpy.linspace(0,6,3)
ciffileV,rpoV,sp0V,LV,xc0V,k0V=numpy.meshgrid(ciffile,rpo,sp0,L,xc0,k0)

colors=['tab:red','tab:green','tab:blue']
superscripts=['100','010','001']
datanames=['rpo','sp0','L','xc0','k0']
datasigns=[r'r_{\rm{po}}/1',r's_p^0/\rm{\AA}^{-1}',r'\overline{L}/\rm{nm}',r'x_{\rm{c}}^0/1',r'k^0/\rm{\AA}^2']
abc=[r'$\rm{(b)}$',r'$\rm{(a)}$',r'$\rm{(d)}$',r'$\rm{(c)}$']

os.system('cp data.npy data_alt.npy')

if len(sys.argv)>1 and sys.argv[1]=='calculate':
	os.system('rm *.txt')
	for c in ciffile:
		phasename=os.path.splitext(c)[0]
		lattice=xu.materials.cif.CIFFile(c).SGLattice()
		crystal=xu.materials.material.Crystal(phasename,lattice)
		f=open(phasename+'_reflections.txt','w')
		f.write(str(xu.simpack.PowderDiffraction(crystal)))
	os.system('rm *.png')
	os.system('rm *.pdf')
	ray.init(num_cpus=os.cpu_count()-1)
	data=numpy.array(ray.get([calc.remote(ciffileV.flatten()[i],rpoV.flatten()[i],sp0V.flatten()[i],LV.flatten()[i],xc0V.flatten()[i],k0V.flatten()[i]) for i,j in enumerate(rpoV.flatten())]),dtype=object)
	numpy.save('data',data)

data=numpy.load('data.npy',allow_pickle=True)

ciffile,rpo,sp0,L,xc0,k0,xc_collect,k_collect=data[:,0],data[:,1].astype('float64'),data[:,2].astype('float64'),data[:,3].astype('float64'),data[:,4].astype('float64'),data[:,5].astype('float64'),data[:,6],data[:,7]

xc100=uq([i[0].magnitude for i in xc_collect],xc_collect[0][0].units,[i[0].uncertainty for i in xc_collect])
xc010=uq([i[1].magnitude for i in xc_collect],xc_collect[0][1].units,[i[1].uncertainty for i in xc_collect])
xc001=uq([i[2].magnitude for i in xc_collect],xc_collect[0][2].units,[i[2].uncertainty for i in xc_collect])
k100=uq([i[0].magnitude for i in k_collect],k_collect[0][0].units,[i[0].uncertainty for i in k_collect])
k010=uq([i[1].magnitude for i in k_collect],k_collect[0][1].units,[i[1].uncertainty for i in k_collect])
k001=uq([i[2].magnitude for i in k_collect],k_collect[0][2].units,[i[2].uncertainty for i in k_collect])

L*=1e9

for c,valuec in enumerate(numpy.unique(ciffile)):
	argsc=numpy.where(ciffile==valuec)

	for i,valuei in enumerate([rpo,sp0,L,xc0,k0]):
		plt.clf()
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		fig,ax1=plt.subplots(figsize=(7.5/2.54,7.5/2.54/2))

		text_collect=[]
		for j,valuej in enumerate([xc100.magnitude,xc010.magnitude,xc001.magnitude]):
			offset=(max(valuei)-min(valuei))*(-1/100+j/100)
			meds_collect=[]
			for valuek in numpy.unique(valuei[argsc]):
				argsk=numpy.where(valuei[argsc]==valuek)
				meds_collect.append(numpy.median((valuej-xc0)[argsc][argsk]))
			ax1.plot(numpy.unique(valuei[argsc])+offset,meds_collect,color=colors[j],linewidth='0.5')

			ax1.scatter(valuei[argsc]+offset,(valuej-xc0)[argsc],marker='s',s=1,edgecolors='none',c=colors[j])
			tau,p=scipy.stats.kendalltau(valuei[argsc],(valuej-xc0)[argsc])

			text_collect.append(r'$\tau^{'+superscripts[j]+'}='+'%.2f'%tau+'$')

		for j,t in enumerate(text_collect):
			ax1.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])*2/3,ax1.get_ylim()[1]-(ax1.get_ylim()[1]-ax1.get_ylim()[0])*(j+1)/10,t,fontsize=6,color=colors[j])

		plt.figtext(0,0.925,abc[c],fontsize=10)

		ax1.set_xlabel(r'$'+datasigns[i]+'$',labelpad=1,fontsize=8)
		ax1.set_ylabel(r'$(x_{\rm{c}}-x_{\rm{c}}^0)/1$',labelpad=2,fontsize=8)
		ax1.tick_params(axis='x',pad=1,labelsize=7)
		ax1.tick_params(axis='y',pad=1,labelsize=7)
		plt.tight_layout(pad=0.1)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+datanames[i]+'_corrxc.png',dpi=300)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+datanames[i]+'_corrxc.pdf',transparent=True)
		plt.close('all')

	for i,valuei in enumerate([rpo,sp0,L,xc0,k0]):
		plt.clf()
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		fig,ax1=plt.subplots(figsize=(7.5/2.54,7.5/2.54/2))

		text_collect=[]
		for j,valuej in enumerate([k100.magnitude,k010.magnitude,k001.magnitude]):
			offset=(max(valuei)-min(valuei))*(-1/100+j/100)
			meds_collect=[]
			for valuek in numpy.unique(valuei[argsc]):
				argsk=numpy.where(valuei[argsc]==valuek)
				meds_collect.append(numpy.median((valuej-k0)[argsc][argsk]))
			ax1.plot(numpy.unique(valuei[argsc])+offset,meds_collect,color=colors[j],linewidth='0.5')

			ax1.scatter(valuei[argsc]+offset,(valuej-k0)[argsc],marker='s',s=1,edgecolors='none',c=colors[j])
			tau,p=scipy.stats.kendalltau(valuei[argsc],(valuej-k0)[argsc])

			text_collect.append(r'$\tau^{'+superscripts[j]+'}='+'%.2f'%tau+'$')

		for j,t in enumerate(text_collect):
			ax1.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])*2/3,ax1.get_ylim()[1]-(ax1.get_ylim()[1]-ax1.get_ylim()[0])*(j+1)/10,t,fontsize=6,color=colors[j])

		plt.figtext(0,0.925,abc[c],fontsize=10)

		ax1.set_xlabel(r'$'+datasigns[i]+'$',labelpad=1,fontsize=8)
		ax1.set_ylabel(r'$(k-k^0)/1$',labelpad=2,fontsize=8)
		ax1.tick_params(axis='x',pad=1,labelsize=7)
		ax1.tick_params(axis='y',pad=1,labelsize=7)
		plt.tight_layout(pad=0.1)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+datanames[i]+'_corrk.png',dpi=300)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+datanames[i]+'_corrk.pdf',transparent=True)
		plt.close('all')

	for critxc in [0.03,0.05,0.1]:
		argsacchkl=numpy.where(	(abs(xc100.magnitude[argsc]-xc0[argsc])<critxc)&\
								(abs(xc010.magnitude[argsc]-xc0[argsc])<critxc)&\
								(abs(xc001.magnitude[argsc]-xc0[argsc])<critxc))
		argsacc001=numpy.where(  abs(xc001.magnitude[argsc]-xc0[argsc])<critxc)

		plt.clf()
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		fig,axs=plt.subplots(1,5,figsize=(7.5/2.54,7.5/2.54/2))
		axt=numpy.empty_like(axs)

		for i,valuei in enumerate([rpo,sp0,L,xc0,k0]):
			vp=axs[i].violinplot([valuei[argsc][argsacchkl],valuei[argsc][argsacc001]],showextrema=False,widths=[len(valuei[argsc][argsacchkl])/len(valuei[argsc]),\
																												 len(valuei[argsc][argsacc001])/len(valuei[argsc])],bw_method=3/4)
			for p,pc in enumerate(vp['bodies']):
				if p%2==0:
					pc.set_facecolor('tab:gray')
				else:
					pc.set_facecolor('tab:blue')
				pc.set_edgecolor('k')
				pc.set_alpha(1)
				pc.set_linewidth(0.5)
			axs[i].set_xticks([1,2])
			axs[i].set_xlim([0.5,2.5])
			axs[i].set_xticklabels([r'$hkl$',r'$001$'])
			axs[i].set_xlabel(r'$'+datasigns[i]+'$',labelpad=1,fontsize=8)
			axs[i].tick_params(axis='x',pad=1,labelsize=7)
			axs[i].tick_params(axis='y',pad=1,labelsize=7)
			axt[i]=axs[i].twiny()
			axt[i].set_xticks([0.25,0.75])
			axt[i].set_xticklabels([r'$'+str(len(valuei[argsc][argsacchkl]))+'$',r'$'+str(len(valuei[argsc][argsacc001]))+'$'])
			axt[i].tick_params(axis='x',pad=0,labelsize=7)
			axs[i].set_ylim([min(valuei)-max(valuei)*0.05,max(valuei)*1.05])

		plt.figtext(0,0.925,abc[c],fontsize=10)

		plt.tight_layout(pad=0.1)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+str(critxc).replace('.','-')+'_accsxc.png',dpi=300)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+str(critxc).replace('.','-')+'_accsxc.pdf',transparent=True)
		plt.close('all')

	for critk in [0.3,0.5,1.0]:
		argsacchkl=numpy.where(	(abs(k100.magnitude[argsc]-k0[argsc])<critk)&\
								(abs(k010.magnitude[argsc]-k0[argsc])<critk)&\
								(abs(k001.magnitude[argsc]-k0[argsc])<critk))
		argsacc001=numpy.where(  abs(k001.magnitude[argsc]-k0[argsc])<critk)

		plt.clf()
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		fig,axs=plt.subplots(1,5,figsize=(7.5/2.54,7.5/2.54/2))
		axt=numpy.empty_like(axs)

		for i,valuei in enumerate([rpo,sp0,L,xc0,k0]):
			vp=axs[i].violinplot([valuei[argsc][argsacchkl],valuei[argsc][argsacc001]],showextrema=False,widths=[len(valuei[argsc][argsacchkl])/len(valuei[argsc]),\
																												 len(valuei[argsc][argsacc001])/len(valuei[argsc])],bw_method=3/4)
			for p,pc in enumerate(vp['bodies']):
				if p%2==0:
					pc.set_facecolor('tab:gray')
				else:
					pc.set_facecolor('tab:blue')
				pc.set_edgecolor('k')
				pc.set_alpha(1)
				pc.set_linewidth(0.5)
			axs[i].set_xticks([1,2])
			axs[i].set_xlim([0.5,2.5])
			axs[i].set_xticklabels([r'$hkl$',r'$001$'])
			axs[i].set_xlabel(r'$'+datasigns[i]+'$',labelpad=1,fontsize=8)
			axs[i].tick_params(axis='x',pad=1,labelsize=7)
			axs[i].tick_params(axis='y',pad=1,labelsize=7)
			axt[i]=axs[i].twiny()
			axt[i].set_xticks([0.25,0.75])
			axt[i].set_xticklabels([r'$'+str(len(valuei[argsc][argsacchkl]))+'$',r'$'+str(len(valuei[argsc][argsacc001]))+'$'])
			axt[i].tick_params(axis='x',pad=0,labelsize=7)
			axs[i].set_ylim([min(valuei)-max(valuei)*0.05,max(valuei)*1.05])

		plt.figtext(0,0.925,abc[c],fontsize=10)

		plt.tight_layout(pad=0.1)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+str(critk).replace('.','-')+'_accsk.png',dpi=300)
		plt.savefig(os.path.splitext(valuec)[0]+'_'+str(critk).replace('.','-')+'_accsk.pdf',transparent=True)
		plt.close('all')

plt.clf()
mpl.rc('text',usetex=True)
mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
fig,ax1=plt.subplots(figsize=(7.5/2.54,6.5/2.54))

corr=numpy.corrcoef([rpo,sp0,L,xc0,k0,	xc100.magnitude-xc0,\
										xc010.magnitude-xc0,\
										xc001.magnitude-xc0,\
										k100.magnitude-k0,\
										k010.magnitude-k0,\
										k001.magnitude-k0])
cax=ax1.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1,interpolation='none')
cbar=fig.colorbar(cax,fraction=0.1,aspect=25,pad=0.02,ticks=numpy.linspace(-1,1,9))
cbar.ax.set_yticklabels([r'-$1.00$',r'-$0.75$',r'-$0.50$',r'-$0.25$',r'$0.00$',r'$0.25$',r'$0.50$',r'$0.75$',r'$1.00$'],fontsize=7)
for (x,y),value in numpy.ndenumerate(corr):
	if value.round(1)<0:
		sign='-'
	else:
		sign=''
	plt.text(x,y,sign+r'$'+'%.1f'%abs(value)+'$',va="center",ha="center",fontsize=6)

cbar.ax.tick_params(labelsize=7)
ticks=numpy.arange(0,11,1)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
corrsigns=[ r'$r_{\rm{po}}$',r'$s_p^0$',r'$\overline{L}$',r'$x_{\rm{c}}^0$',r'$k^0$',\
			r'$x_{\rm{c}}^{100}$-$x_{\rm{c}}^0$',\
			r'$x_{\rm{c}}^{010}$-$x_{\rm{c}}^0$',\
			r'$x_{\rm{c}}^{001}$-$x_{\rm{c}}^0$',\
			r'$k^{100}$-$k^0$',\
			r'$k^{010}$-$k^0$',\
			r'$k^{001}$-$k^0$']
ax1.set_xticklabels(corrsigns)
ax1.set_yticklabels(corrsigns)
plt.xticks(rotation=90)
ax1.tick_params(axis='x',pad=1,labelsize=8)
ax1.tick_params(axis='y',pad=1,labelsize=8)
plt.tight_layout(pad=0.1)
plt.savefig('corr.png',dpi=300)
plt.savefig('corr.pdf',transparent=True)
plt.close('all')
