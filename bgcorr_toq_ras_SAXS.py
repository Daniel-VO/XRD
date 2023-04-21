import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

paths=glob.glob('*/')
paths.append('')

for p in paths:
	BGfiles=glob.glob(p+'*BG*.ras')

	if len(BGfiles)>1:
		print('Warnung: Mehr als ein Untergrund!')
	elif len(BGfiles)==1:
		tt_bg,yobs_bg,yerr_bg=np.genfromtxt((i.replace('*','#') for i in open(BGfiles[0])),unpack=True)
		argsavg=np.where((tt_bg>=min(tt_bg))&(tt_bg<=-min(tt_bg)))
		tt_bg-=np.average(tt_bg[argsavg],weights=yobs_bg[argsavg]**2)				#zero drift correction
		f=interpolate.interp1d(tt_bg,yobs_bg)

	for i in glob.glob(p+'*[!BG].ras'):
		filename=os.path.splitext(i)[0]
		print(filename)
		tt,yobs,yerr=np.genfromtxt((i.replace('*','#') for i in open(i)),unpack=True)
		if min(tt)<0:
			argsavg=np.where((tt>=min(tt))&(tt<=-min(tt)))
			tt-=np.average(tt[argsavg],weights=yobs[argsavg]**2)					#zero drift correction
		if len(BGfiles)==1:
			argscut=np.where((tt>=min(tt_bg))&(tt<=max(tt_bg)))
			tt=tt[argscut];yobs=yobs[argscut]
			yobs_bg=f(tt)
			yobs-=yobs_bg															#bgcorr

		q=4*np.pi*np.sin(np.radians(tt/2))/0.15406									#toq
		mincoord=np.where(yobs==max(yobs[np.where(q>0)]))[0][0]
		print('qmin = '+str(q[mincoord])+' nm^-1')

		plt.close('all')
		if len(BGfiles)==1:
			plt.plot(q,yobs+yobs_bg);plt.plot(q,yobs_bg)
		plt.plot(q,yobs)
		plt.yscale('log'),plt.xlim([-0.01,0.01])
		plt.savefig(filename+'_cb.png')

		plt.close('all')
		if len(BGfiles)==1:
			plt.plot(q,yobs+yobs_bg);plt.plot(q,yobs_bg)
		plt.plot(q,yobs);plt.plot(q[mincoord:],yobs[mincoord:])
		plt.xscale('log'),plt.yscale('log'),plt.xlim([1e-3,None])
		plt.savefig(filename+'.png')

		np.savetxt(filename+'_zd_bg_tq.dat',np.transpose([q[mincoord:],yobs[mincoord:]]),fmt='%.8f')

